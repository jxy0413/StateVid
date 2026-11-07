"""Command-line interfaces for StateVid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ChangeDataset, collate_batch, decode_caption, load_manifest, normalize_image
from .engine import evaluate_model, load_checkpoint, load_config, select_device, train, write_json
from .models import build_model


def _common_device(parser):
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N; default selects CUDA when available")
    parser.add_argument("--threads", type=int, default=4, help="CPU tensor worker threads")


def _train_parser(parser):
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    _common_device(parser)


def _evaluate_parser(parser):
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--caption-backend", choices=("builtin", "coco"), default=None)
    parser.add_argument("--include-meteor", action="store_true")
    _common_device(parser)


def _predict_parser(parser):
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pre", type=Path)
    parser.add_argument("--post", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--tile-size", type=int)
    _common_device(parser)


def _inspect_parser(parser):
    parser.add_argument("--task", choices=("bcd", "scd", "bda", "cc"), default="bcd")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--vocab-size", type=int, default=404)
    parser.add_argument("--depth-factor", type=float, default=5)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    _common_device(parser)


def run_train(args):
    torch.set_num_threads(args.threads)
    return train(load_config(args.config), args.output, device=args.device,
                 max_steps=args.max_steps, resume=args.resume)


def run_evaluate(args):
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model, saved = load_checkpoint(args.checkpoint, device)
    load_manifest(args.manifest, model.task)
    caption = saved.get("config", {}).get("caption", {})
    dataset = ChangeDataset(args.manifest, args.split, augment=False, wordmap=saved.get("wordmap"),
                             max_words=model.config["max_length"]-2, tile_size=args.tile_size,
                             cc_image_size=int(caption.get("image_size", 256)))
    result = evaluate_model(model, dataset, device=device, batch_size=args.batch_size,
                            workers=args.workers, wordmap=saved.get("wordmap"),
                            beam_size=caption.get("beam_size", 3),
                            caption_backend=args.caption_backend or caption.get("metric_backend", "builtin"),
                            include_meteor=args.include_meteor)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return result


def _save_dense(output, maps):
    output.mkdir(parents=True, exist_ok=True)
    for name, value in maps.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        Image.fromarray(value.astype(np.uint8)).save(output / (name + ".png"))


@torch.inference_mode()
def _predict_one(model, pre, post, output, size, wordmap, beam_size):
    h, w = size
    output.mkdir(parents=True, exist_ok=True)
    if model.task == "cc":
        if wordmap is None:
            raise ValueError("Caption checkpoint is missing its training vocabulary")
        ids = model.generate(pre, post, wordmap["<start>"], wordmap["<end>"], beam_size=beam_size)[0]
        words = decode_caption(ids, wordmap)
        write_json(output / "caption.json", {"caption": " ".join(words), "tokens": words})
        (output / "caption.txt").write_text(" ".join(words) + "\n", encoding="utf-8")
        return
    pad_h, pad_w = max(16, (h+7)//8*8), max(16, (w+7)//8*8)
    pre, post = [F.pad(x, (0, pad_w-w, 0, pad_h-h), mode="replicate") for x in (pre, post)]
    result = model(pre, post)
    if model.task == "bcd":
        maps = {"change_mask": (result["change_logits"][0,0,:h,:w].sigmoid() > .5).byte()*255}
    elif model.task == "scd":
        change = result["pred_cd"][0,0,:h,:w] > .5
        maps = {"pre_labels": result["pred_a"][0,:,:h,:w].argmax(0)*change,
                "post_labels": result["pred_b"][0,:,:h,:w].argmax(0)*change,
                "change_mask": change.byte()*255}
    else:
        localization = result["localization_logits"][0,0,:h,:w].sigmoid() > .5
        damage = result["damage_logits"][0,:,:h,:w].argmax(0)
        maps = {"localization_mask": localization.byte()*255, "damage_labels": damage,
                "damage_map": damage*localization}
    _save_dense(output, maps)


def run_predict(args):
    torch.set_num_threads(args.threads)
    if bool(args.manifest) == bool(args.pre or args.post):
        raise ValueError("Provide either --manifest or both --pre and --post")
    if not args.manifest and not (args.pre and args.post):
        raise ValueError("Single-pair prediction needs both --pre and --post")
    device = select_device(args.device)
    model, saved = load_checkpoint(args.checkpoint, device)
    model.eval()
    wordmap = saved.get("wordmap")
    caption = saved.get("config", {}).get("caption", {})
    if args.manifest:
        load_manifest(args.manifest, model.task)
        dataset = ChangeDataset(args.manifest, args.split, augment=False, require_labels=False,
                                 max_words=model.config["max_length"]-2, tile_size=args.tile_size,
                                 cc_image_size=int(caption.get("image_size", 256)))
        count = 0
        for batch in DataLoader(dataset, batch_size=1, collate_fn=collate_batch):
            if args.limit is not None and count >= args.limit:
                break
            identifier = re.sub(r"[^A-Za-z0-9_.-]", "_", batch["id"][0])
            # Include the sequence position to prevent colliding sanitized IDs.
            destination = args.output / f"{count:06d}_{identifier}"
            h, w = batch["original_size"][0]
            pre, post = [(batch[k] if model.task == "cc" else batch[k][:,:,:h,:w]).to(device)
                         for k in ("pre", "post")]
            _predict_one(model, pre, post, destination,
                         pre.shape[-2:] if model.task == "cc" else (h,w), wordmap, args.beam_size)
            count += 1
        print(json.dumps({"predicted_pairs": count, "output": str(args.output)}))
    else:
        images = [Image.open(p).convert("RGB") for p in (args.pre, args.post)]
        if images[0].size != images[1].size:
            raise ValueError("Paired input images must share their dimensions")
        w, h = images[0].size
        if model.task == "cc":
            image_size = int(caption.get("image_size", 256))
            images = [x.resize((image_size,image_size), Image.Resampling.BILINEAR) for x in images]
            h, w = image_size, image_size
        pre, post = [normalize_image(np.asarray(x), model.task)[None].to(device) for x in images]
        if args.tile_size is not None:
            raise ValueError("--tile-size is available with manifest prediction; omit it for a single native pair")
        _predict_one(model, pre, post, args.output, (h,w), wordmap, args.beam_size)
        print(json.dumps({"predicted_pairs": 1, "output": str(args.output)}))


def run_inspect(args):
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    model = build_model(task=args.task, num_classes=args.num_classes,
                         vocab_size=args.vocab_size if args.task == "cc" else None,
                         depth_factor=args.depth_factor, grid_size=args.grid_size).to(device).eval()
    image = torch.zeros(1,3,args.input_size,args.input_size,device=device)
    with torch.inference_mode():
        output = (model(image,image,torch.ones(1,2,dtype=torch.long,device=device))
                  if args.task == "cc" else model(image,image))
    report = dict(model.parameter_report(), outputs={k:list(v.shape) for k,v in output.items()})
    if args.output:
        write_json(args.output,report)
    print(json.dumps(report,indent=2))
    return report


def _entry(builder, runner):
    parser = argparse.ArgumentParser(description=__doc__)
    builder(parser)
    return runner(parser.parse_args())


def train_main():
    _entry(_train_parser, run_train)


def evaluate_main():
    _entry(_evaluate_parser, run_evaluate)


def predict_main():
    _entry(_predict_parser, run_predict)


def inspect_main():
    _entry(_inspect_parser, run_inspect)


def main():
    parser = argparse.ArgumentParser(description="StateVid training, evaluation, and prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    actions = {"train": (_train_parser, run_train), "evaluate": (_evaluate_parser, run_evaluate),
               "predict": (_predict_parser, run_predict), "inspect": (_inspect_parser, run_inspect)}
    for name, (builder, _) in actions.items():
        builder(commands.add_parser(name))
    args = parser.parse_args()
    return actions[args.command][1](args)


if __name__ == "__main__":
    main()
