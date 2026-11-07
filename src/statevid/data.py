"""Portable image-pair manifests, synchronized geometry, and caption vocabularies."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

TASKS = ("bcd", "scd", "bda", "cc")
SPECIAL_TOKENS = ("<pad>", "<unk>", "<start>", "<end>")
IMAGENET_MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
IMAGENET_STD = np.asarray([.229, .224, .225], dtype=np.float32)


def load_manifest(path: str | Path, task: str | None = None) -> dict[str, Any]:
    """Read a manifest without opening or downloading its image files."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("task") not in TASKS:
        raise ValueError(f"Manifest task must be one of {TASKS}")
    if task is not None and data["task"] != task:
        raise ValueError("Manifest task does not match the requested model")
    if not isinstance(data.get("splits"), dict) or not data["splits"]:
        raise ValueError("Manifest needs a nonempty 'splits' mapping")
    seen_pairs: dict[tuple[str, str], str] = {}
    base = Path(path).expanduser().resolve().parent
    root = Path(data.get("root", ".")).expanduser()
    root = root if root.is_absolute() else base / root
    for split, records in data["splits"].items():
        if not isinstance(records, list):
            raise ValueError(f"Split {split!r} must be a list")
        ids: set[str] = set()
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
                raise ValueError("Every record needs a nonempty string id")
            if record["id"] in ids:
                raise ValueError(f"Duplicate id {record['id']!r} within {split!r}")
            ids.add(record["id"])
            for key in ("pre", "post"):
                if not isinstance(record.get(key), str) or not record[key]:
                    raise ValueError(f"Record {record['id']!r} needs a {key!r} image path")
            pair = tuple(sorted(str((root / record[key]).resolve()) for key in ("pre", "post")))
            if pair in seen_pairs and seen_pairs[pair] != split:
                raise ValueError("An identical or reversed image pair occurs in different splits")
            seen_pairs[pair] = split
    if data["task"] == "scd" and (type(data.get("num_classes")) is not int or data["num_classes"] < 2):
        raise ValueError("SCD requires num_classes, including unchanged class 0")
    return data


def resolve_path(manifest_path: str | Path, value: str, root: str = ".") -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path(root).expanduser()
    if not base.is_absolute():
        base = Path(manifest_path).expanduser().resolve().parent / base
    return base / path


def caption_tokens(value: str | list[str] | dict) -> list[str]:
    """Keep supplied tokens, or split a supplied sentence at whitespace."""
    if isinstance(value, dict):
        value = value.get("tokens", value.get("text"))
    tokens = value.split() if isinstance(value, str) else value
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError("Each caption must be a nonempty sentence or list of tokens")
    if any(not isinstance(x, str) or not x or any(c.isspace() for c in x) for x in tokens):
        raise ValueError("Caption tokens must be nonempty strings without whitespace")
    if any(x in SPECIAL_TOKENS for x in tokens):
        raise ValueError("References must not contain decoder boundary/padding tokens")
    return list(tokens)


def validate_wordmap(wordmap: dict[str, int] | str | Path) -> dict[str, int]:
    if isinstance(wordmap, (str, Path)):
        wordmap = json.loads(Path(wordmap).read_text(encoding="utf-8"))
    if not isinstance(wordmap, dict) or any(token not in wordmap for token in SPECIAL_TOKENS):
        raise ValueError("Vocabulary needs <pad>, <unk>, <start>, and <end>")
    if any(not isinstance(k, str) or type(v) is not int for k, v in wordmap.items()):
        raise ValueError("Vocabulary must map strings to integer ids")
    if wordmap["<pad>"] != 0 or set(wordmap.values()) != set(range(len(wordmap))):
        raise ValueError("Vocabulary ids must be contiguous and unique, with <pad>=0")
    return dict(wordmap)


def build_wordmap(manifest: str | Path | dict, min_word_freq: int = 5,
                  max_words: int = 50) -> tuple[dict[str, int], dict]:
    """Build the vocabulary from training captions, using count > threshold."""
    data = load_manifest(manifest, "cc") if not isinstance(manifest, dict) else manifest
    if data.get("task") != "cc" or not data.get("splits", {}).get("train"):
        raise ValueError("Caption vocabulary construction requires a nonempty training split")
    if type(min_word_freq) is not int or min_word_freq < 0 or type(max_words) is not int or max_words < 1:
        raise ValueError("Invalid vocabulary threshold or maximum caption length")
    counts: Counter = Counter()
    for record in data["splits"]["train"]:
        references = record.get("captions")
        if not isinstance(references, list) or not references:
            raise ValueError("Every captioning record needs a list of references")
        for sentence in references:
            tokens = caption_tokens(sentence)
            if len(tokens) > max_words:
                raise ValueError("A training caption exceeds max_words; increase the configured length")
            counts.update(tokens)
    words = [word for word, count in counts.items() if count > min_word_freq]
    result = {word: index+1 for index, word in enumerate(words)}
    for token in ("<unk>", "<start>", "<end>"):
        result[token] = len(result)+1
    result["<pad>"] = 0
    return result, {"source": "train", "min_word_freq": min_word_freq,
                    "threshold": "count > min_word_freq", "max_words": max_words,
                    "vocab_size": len(result)}


def encode_caption(tokens: str | list[str], wordmap: dict[str, int],
                   max_words: int = 50) -> tuple[torch.Tensor, int]:
    tokens = caption_tokens(tokens)
    if len(tokens) > max_words:
        raise ValueError("Caption exceeds max_words; increase the configured sequence length")
    values = [wordmap["<start>"]] + [wordmap.get(x, wordmap["<unk>"]) for x in tokens] + [wordmap["<end>"]]
    length = len(values)
    values += [wordmap["<pad>"]] * (max_words+2-length)
    return torch.tensor(values, dtype=torch.long), length


def decode_caption(ids, wordmap: dict[str, int]) -> list[str]:
    inverse = {value: key for key, value in validate_wordmap(wordmap).items()}
    words = []
    for value in ids:
        word = inverse.get(int(value))
        if word is None:
            raise ValueError("Generated token id is outside the vocabulary")
        if word == "<end>":
            break
        if word not in ("<pad>", "<start>"):
            words.append(word)
    return words


def normalize_image(array: np.ndarray, task: str, normalization: str | None = None) -> torch.Tensor:
    value = np.asarray(array, dtype=np.float32)/255.
    mode = normalization or ("imagenet" if task == "cc" else "minus_one_one")
    if mode in ("minus_one_one", "symmetric"):
        value = (value-.5)/.5
    elif mode == "imagenet":
        value = (value-IMAGENET_MEAN)/IMAGENET_STD
    elif mode != "none":
        raise ValueError("normalization must be minus_one_one, imagenet, or none")
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))


def decode_index_mask(array: np.ndarray, num_classes: int, palette=None,
                      ignore_index: int | None = 255) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(array)
    if array.ndim == 3:
        if array.shape[2] not in (3, 4) or palette is None:
            raise ValueError("RGB semantic masks require an explicit palette_rgb")
        result = np.full(array.shape[:2], -1, dtype=np.int64)
        for index, color in enumerate(palette):
            result[np.all(array[:, :, :3] == np.asarray(color), axis=-1)] = index
        if (result < 0).any():
            raise ValueError("Semantic label contains a color absent from palette_rgb")
    elif array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
        result = array.astype(np.int64, copy=True)
    else:
        raise ValueError("Semantic labels must be indexed HW masks or exact RGB palette masks")
    valid = np.ones(result.shape, dtype=bool) if ignore_index is None else result != ignore_index
    if ((result[valid] < 0) | (result[valid] >= num_classes)).any():
        raise ValueError("Semantic labels are outside 0..num_classes-1")
    result[~valid] = 0
    return result, valid


def decode_binary_mask(array: np.ndarray, ignore_index: int | None = None,
                       positive_value: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        if not np.all(array[:, :, :3] == array[:, :, :1]):
            raise ValueError("Binary RGB masks must have equal color channels")
        array = array[:, :, 0]
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Binary masks must be integer HW arrays")
    valid = np.ones(array.shape, bool) if ignore_index is None else array != ignore_index
    values = set(np.unique(array[valid]).tolist())
    if positive_value is None:
        if values.issubset({0, 1}):
            positive_value = 1
        elif values.issubset({0, 255}):
            positive_value = 255
        else:
            raise ValueError("Binary labels must use 0/1 or 0/255; configure ignore_index for ignored pixels")
    if positive_value == ignore_index:
        raise ValueError("Positive and ignored binary values must differ")
    if not values.issubset({0, positive_value}):
        raise ValueError("Binary mask contains unexpected label values")
    return ((array == positive_value) & valid).astype(np.float32), valid


class ChangeDataset(Dataset):
    """Read any of the four tasks through the same manifest and batch schema.

    Dense evaluation retains native image sizes unless ``tile_size`` is set.
    Training applies synchronized crops and lossless rotations/flips. Caption
    training visits each reference; evaluation visits each image pair once.
    """
    def __init__(self, manifest_path: str | Path, split: str,
                 crop_size: int | None = 256, augment: bool | None = None,
                 wordmap: dict | str | Path | None = None, max_words: int = 50,
                 cc_image_size: int = 256, tile_size: int | None = None,
                 require_labels: bool = True):
        self.path = Path(manifest_path).expanduser().resolve()
        self.manifest = load_manifest(self.path)
        self.task = self.manifest["task"]
        if split not in self.manifest["splits"]:
            raise ValueError(f"Manifest has no {split!r} split")
        self.records = self.manifest["splits"][split]
        self.split, self.crop_size = split, crop_size
        self.augment = (split == "train") if augment is None else bool(augment)
        if self.augment and split != "train":
            raise ValueError("Geometric augmentation is available only for the training split")
        self.wordmap = validate_wordmap(wordmap) if wordmap is not None else None
        self.max_words, self.cc_image_size = max_words, cc_image_size
        self.tile_size, self.require_labels = tile_size, require_labels
        if self.task == "cc" and require_labels and self.wordmap is None:
            raise ValueError("Caption datasets require a training vocabulary")
        if crop_size is not None and (type(crop_size) is not int or crop_size < 1):
            raise ValueError("crop_size must be a positive integer or None")
        if tile_size is not None and (type(tile_size) is not int or tile_size < 1):
            raise ValueError("tile_size must be a positive integer or None")
        self.entries = []
        for i, record in enumerate(self.records):
            if self.task == "cc":
                refs = record.get("captions", [])
                if require_labels and not refs:
                    raise ValueError("Every caption pair needs at least one reference")
                refs = [caption_tokens(x) for x in refs]
                if any(len(x) > max_words for x in refs):
                    raise ValueError("A reference exceeds max_words; increase the configured sequence length")
                count = len(refs) if split == "train" and require_labels else 1
                self.entries.extend((i, j, None) for j in range(count))
            elif tile_size is not None:
                with Image.open(self._path(record["pre"])) as image:
                    w, h = image.size
                self.entries.extend((i, None, (top, left)) for top in range(0, h, tile_size)
                                    for left in range(0, w, tile_size))
            else:
                self.entries.append((i, None, None))

    def _path(self, value: str) -> Path:
        return resolve_path(self.path, value, self.manifest.get("root", "."))

    def _read(self, record, key, rgb=False):
        if not isinstance(record.get(key), str):
            raise ValueError(f"Record {record['id']!r} needs label path {key!r}")
        with Image.open(self._path(record[key])) as image:
            return np.asarray(image.convert("RGB") if rgb else image).copy()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        rec_index, caption_index, tile = self.entries[index]
        record = self.records[rec_index]
        pre, post = self._read(record, "pre", True), self._read(record, "post", True)
        if pre.shape != post.shape:
            raise ValueError("Pre- and post-change RGB images must have matching shapes")
        original_size = tuple(pre.shape[:2])
        result = {"id": record["id"], "source_id": record["id"], "task": self.task}
        if self.task == "cc":
            size = (self.cc_image_size, self.cc_image_size)
            images = [np.asarray(Image.fromarray(x).resize(size, Image.Resampling.BILINEAR))
                      if x.shape[:2] != size[::-1] else x for x in (pre, post)]
            result.update(pre=normalize_image(images[0], self.task, self.manifest.get("normalization")),
                          post=normalize_image(images[1], self.task, self.manifest.get("normalization")),
                          original_size=original_size,
                          valid_mask=torch.ones(size[::-1], dtype=torch.bool))
            if self.require_labels:
                refs = [caption_tokens(x) for x in record["captions"]]
                tokens, length = encode_caption(refs[caption_index], self.wordmap, self.max_words)
                result.update(tokens=tokens, length=length, references=refs,
                              caption_index=caption_index, changeflag=record.get("changeflag", -1))
            return result
        valid = np.ones(pre.shape[:2], dtype=bool)
        if record.get("valid_mask"):
            mask = self._read(record, "valid_mask")
            if mask.ndim != 2 or mask.shape != valid.shape:
                raise ValueError("valid_mask must be a matching grayscale image")
            valid &= mask > 0
        values = {"pre": pre, "post": post}
        if self.require_labels and self.task == "bcd":
            label, usable = decode_binary_mask(self._read(record, "label"),
                self.manifest.get("binary_ignore_index"), self.manifest.get("binary_positive_value"))
            values["label"] = label
            valid &= usable
        elif self.require_labels and self.task == "scd":
            for key in ("label_pre", "label_post"):
                label, usable = decode_index_mask(self._read(record, key), self.manifest["num_classes"],
                    self.manifest.get("palette_rgb"), self.manifest.get("ignore_index", 255))
                values[key] = label
                valid &= usable
            if not np.array_equal((values["label_pre"] > 0)[valid], (values["label_post"] > 0)[valid]):
                raise ValueError("SCD endpoint masks must share the same changed region (nonzero labels)")
            if record.get("pseudo_pre") or record.get("pseudo_post"):
                if not record.get("pseudo_pre") or not record.get("pseudo_post"):
                    raise ValueError("Cached pseudo labels need both pseudo_pre and pseudo_post")
                for key in ("pseudo_pre", "pseudo_post"):
                    pseudo, usable = decode_index_mask(self._read(record,key),self.manifest["num_classes"],
                        self.manifest.get("palette_rgb"),255)
                    pseudo[~usable] = 0
                    values[key] = pseudo
        elif self.require_labels and self.task == "bda":
            loc, loc_valid = decode_binary_mask(self._read(record, "localization"),
                self.manifest.get("localization_ignore_index", 255),
                self.manifest.get("localization_positive_value", 1))
            damage, damage_valid = decode_index_mask(self._read(record, "damage"), 5,
                self.manifest.get("palette_rgb"), self.manifest.get("damage_ignore_index", 255))
            loc = loc.astype(np.int64)
            loc[~loc_valid] = 255
            damage[~damage_valid] = 255
            values.update(localization=loc, damage=damage)
        values["valid_mask"] = valid
        if any(x.shape[:2] != pre.shape[:2] for x in values.values()):
            raise ValueError("RGB images, labels, and validity masks must share their spatial shape")
        if tile is not None:
            top, left = tile
            values = {k: x[top:top+self.tile_size, left:left+self.tile_size] for k, x in values.items()}
            result.update(id=f"{record['id']}:r{top//self.tile_size}c{left//self.tile_size}",
                          crop_xyxy=(left, top, min(left+self.tile_size, original_size[1]),
                                     min(top+self.tile_size, original_size[0])))
        if self.augment:
            crop = self.crop_size
            if crop is not None:
                h, w = values["pre"].shape[:2]
                if h < crop or w < crop:
                    values = {k: np.pad(x, ((0, max(0,crop-h)), (0,max(0,crop-w))) +
                        (((0,0),) if x.ndim == 3 else ()), constant_values=False if k == "valid_mask" else
                        (255 if k in ("localization", "damage") else 0)) for k, x in values.items()}
                    h, w = max(h,crop), max(w,crop)
                top, left = int(torch.randint(h-crop+1, ()).item()), int(torch.randint(w-crop+1, ()).item())
                values = {k: x[top:top+crop, left:left+crop] for k,x in values.items()}
            rotations = int(torch.randint(4, ()).item())
            horizontal, vertical = bool(torch.randint(2, ()).item()), bool(torch.randint(2, ()).item())
            for key, array in values.items():
                array = np.rot90(array, rotations)
                if horizontal:
                    array = np.flip(array, 1)
                if vertical:
                    array = np.flip(array, 0)
                values[key] = np.ascontiguousarray(array)
        result["original_size"] = tuple(values["pre"].shape[:2])
        for key, array in values.items():
            if key in ("pre", "post"):
                result[key] = normalize_image(array, self.task, self.manifest.get("normalization"))
            elif key == "label":
                result[key] = torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)
            else:
                result[key] = torch.from_numpy(np.ascontiguousarray(array)).bool() if key == "valid_mask" else torch.from_numpy(np.ascontiguousarray(array)).long()
        return result


def collate_batch(batch: list[dict], multiple: int = 8) -> dict:
    """Pad native dense inputs; validity masks exclude every padded pixel."""
    if not batch or len({row["task"] for row in batch}) != 1:
        raise ValueError("A batch must contain samples from one task")
    if multiple < 1:
        raise ValueError("Padding multiple must be positive")
    h = max(row["pre"].shape[-2] for row in batch)
    w = max(row["pre"].shape[-1] for row in batch)
    h, w = (h+multiple-1)//multiple*multiple, (w+multiple-1)//multiple*multiple
    result = {"task": batch[0]["task"], "id": [row["id"] for row in batch],
              "source_id": [row["source_id"] for row in batch],
              "original_size": [row["original_size"] for row in batch]}
    spatial = ("pre", "post", "label", "label_pre", "label_post", "pseudo_pre", "pseudo_post", "localization", "damage", "valid_mask")
    for key in spatial:
        if key not in batch[0]:
            continue
        values = []
        for row in batch:
            array = row[key]
            fill = 255 if key in ("localization", "damage") else 0
            values.append(F.pad(array, (0,w-array.shape[-1],0,h-array.shape[-2]), value=fill))
        result[key] = torch.stack(values)
    for key in ("tokens", "length", "caption_index", "changeflag"):
        if key in batch[0]:
            values = [row[key] for row in batch]
            result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values, dtype=torch.long)
    if "references" in batch[0]:
        result["references"] = [row["references"] for row in batch]
    if "crop_xyxy" in batch[0]:
        result["crop_xyxy"] = [row["crop_xyxy"] for row in batch]
    return result
