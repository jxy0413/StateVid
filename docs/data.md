# Dataset preparation

All tasks use a JSON manifest containing named splits and image-pair records. The configuration's `manifest` field points to this file. Paths inside a manifest are resolved relative to its directory, or relative to the optional manifest `root` field.

## Common structure

```json
{
  "dataset": "LEVIR-CD",
  "task": "bcd",
  "root": ".",
  "splits": {
    "train": [
      {
        "id": "train_001",
        "pre": "train/A/001.png",
        "post": "train/B/001.png",
        "label": "train/label/001.png"
      }
    ],
    "val": [
      {
        "id": "val_001",
        "pre": "val/A/001.png",
        "post": "val/B/001.png",
        "label": "val/label/001.png"
      }
    ],
    "test": [
      {
        "id": "test_001",
        "pre": "test/A/001.png",
        "post": "test/B/001.png",
        "label": "test/label/001.png"
      }
    ]
  }
}
```

Use a unique `id` within each split. The two RGB images and all pixel annotations for a record must share the same spatial dimensions. An optional grayscale `valid_mask` marks usable pixels with a nonzero value and unannotated pixels with zero.

The loader applies the same crop, rotation, and flips to both images and every dense target during training. Dense evaluation uses native sizes, with batch padding to a multiple of eight. Padded pixels are excluded by the validity mask. A configured evaluation tile size divides large scenes into native, non-overlapping tiles.

### Generate a manifest from paired folders

`scripts/prepare_manifest.py` matches filenames by stem within each split and writes paths relative to the output manifest. The default split names are `train`, `val`, and `test`.

```bash
python scripts/prepare_manifest.py --task bcd --dataset LEVIR-CD \
  --root data/levir_cd --output data/levir_cd/manifest.json

python scripts/prepare_manifest.py --task scd --dataset SECOND --num-classes 7 \
  --root data/second --output data/second/manifest.json

python scripts/prepare_manifest.py --task bda --dataset xBD \
  --root data/xbd --output data/xbd/manifest.json
```

The default paired subfolders are `A/B/label` for BCD, `im1/im2/label1/label2` for SCD, and `pre/post/localization/damage` for BDA. Use options such as `--pre-dir`, `--label-pre-dir`, or `--damage-dir` to use different folder names. `--metadata` accepts a JSON file containing class names, a palette, or normalization settings.

For LEVIR-style caption annotations with an `images` list and split names:

```bash
python scripts/prepare_manifest.py --task cc --dataset LEVIR-CC \
  --root data/levir_cc --image-prefix images \
  --captions-json data/levir_cc/LevirCCcaptions.json \
  --output data/levir_cc/manifest.json
```

Each annotation entry supplies `filename`, `split`, and either `captions` or `sentences`. The helper preserves all supplied references and accepts token dictionaries from the original LEVIR-style format.

## Binary change detection

Set `task` to `bcd` and supply `label` in each record. Indexed masks can use `0/1` or `0/255`: zero means unchanged, and the nonzero value means changed. Grayscale RGB copies with equal channels are also accepted.

Optional manifest fields are:

| Field | Meaning |
|---|---|
| `binary_positive_value` | Explicit positive value, such as `1` or `255` |
| `binary_ignore_index` | Value to ignore when a mask contains unannotated pixels |
| `normalization` | Image normalization; defaults to `minus_one_one` |

The positive and ignored values must differ.

## Semantic change detection

Set `task` to `scd`, provide `num_classes` including unchanged class zero, and supply two endpoint label maps:

```json
{
  "dataset": "SECOND",
  "task": "scd",
  "num_classes": 7,
  "class_names": ["Unchanged", "Water", "Ground", "Low vegetation", "Tree", "Building", "Sports field"],
  "ignore_index": 255,
  "splits": {
    "train": [
      {
        "id": "train_001",
        "pre": "train/im1/001.png",
        "post": "train/im2/001.png",
        "label_pre": "train/label1/001.png",
        "label_post": "train/label2/001.png"
      }
    ],
    "val": [],
    "test": []
  }
}
```

Populate `val` and `test` in the same format. Both endpoint maps use zero outside annotated change regions. At every valid pixel, their labels must be either both zero or both nonzero.

Indexed PNG masks use class IDs directly. For RGB annotations, add `palette_rgb`, an ordered list of exact RGB colors for IDs `0` through `num_classes - 1`. The palette defines the numeric mapping; annotations are never reordered from their display colors implicitly.

The supplied dataset configurations use these label spaces:

| Dataset | `num_classes` | IDs, starting at 0 |
|---|---:|---|
| SECOND | 7 | Unchanged, Water, Ground, Low vegetation, Tree, Building, Sports field |
| Landsat-SCD | 5 | Unchanged, Farmland, Desert, Building, Water |
| JL1H | 6 | Unchanged, Cropland, Road, Forest/grassland, Building, Other |

For source annotations with a different numeric ordering, convert the indexed masks to the chosen ordering and use that ordering consistently across all splits. Landsat-SCD can provide a `valid_mask` to exclude its unannotated image footprint.

### Convert label encodings

The label utility can prepare raw annotations before generating the paired-folder manifest.

For JL1H's ordered transition labels, create the two semantic endpoint maps:

```bash
python scripts/convert_labels.py jl1h \
  --input data/jl1h/raw/train/transitions \
  --output-pre data/jl1h/train/label1 \
  --output-post data/jl1h/train/label2
```

Transition IDs `1–4` represent Cropland → Road / Forest-grassland / Building / Other; IDs `5–8` represent the corresponding reverse transitions back to Cropland. ID `0` remains unchanged, and `255` remains ignored in both endpoint maps. Repeat the conversion for the other splits.

For S2Looking's two building-change masks, form their binary union:

```bash
python scripts/convert_labels.py binary-union \
  --first data/s2looking/raw/train/label1 \
  --second data/s2looking/raw/train/label2 \
  --output data/s2looking/train/label
```

Convert an indexed semantic mask with a JSON mapping from old indices to new indices:

```bash
python scripts/convert_labels.py remap \
  --input data/source/labels --output data/prepared/labels \
  --mapping data/label_mapping.json
```

The mapping file can contain an array of destination indices. For RGB label maps, use `--palette data/palette.json` instead; its ordered RGB list defines the output class IDs. The utility writes indexed PNG files and preserves relative subfolders when `--recursive` is used. `--pattern '*.png'` can restrict the input file pattern.

### Cached pseudo labels

SCD records can additionally point to externally prepared pseudo-label maps:

```json
{
  "id": "train_001",
  "pre": "train/im1/001.png",
  "post": "train/im2/001.png",
  "label_pre": "train/label1/001.png",
  "label_post": "train/label2/001.png",
  "pseudo_pre": "pseudo/pre/001.png",
  "pseudo_post": "pseudo/post/001.png"
}
```

These maps use the same class IDs as the semantic targets. Set `train.pseudo_weight` to the desired auxiliary weight. The auxiliary term accepts pixels whose two ground-truth endpoint labels are zero and whose two cached pseudo labels agree on a nonzero class. Set rejected pseudo-label pixels to zero or the ignored value. Prepare the maps with your chosen teacher and confidence policy before training.

The supplied cache utility screens externally generated teacher probabilities. Each NPZ contains normalized `pre` and `post` arrays of shape `K × H × W`. Point a probability manifest to these files:

```json
{
  "samples": {
    "train_001": "probabilities/train_001.npz"
  }
}
```

Then create the cached masks and a new manifest:

```bash
python scripts/cache_pseudo_labels.py \
  --manifest data/second/manifest.json \
  --probabilities data/second/teacher_probabilities.json \
  --cache-dir data/second/pseudo \
  --output data/second/manifest_with_pseudo.json
```

Screening uses cumulative, pixel-count-weighted classwise confidence thresholds derived from the 60th percentile, starting at `0.6` and clipped to `[0.5, 0.9]`. Both endpoints must predict the same nonzero class and pass their thresholds at a valid ground-truth unchanged pixel. The cache utility writes the accepted masks and the threshold accumulator state. Use the new manifest for the training configuration.

When `train.pseudo_weight` is positive, the default training schedule holds the auxiliary weight at zero for ten epochs, then ramps to the configured maximum over five epochs. Set `pseudo_warmup_epochs` and `pseudo_ramp_epochs` to change the schedule, or `pseudo_schedule` to `false` for a constant weight.

## Building damage assessment

Set `task` to `bda` and provide `localization` and `damage` masks:

```json
{
  "id": "disaster_001",
  "pre": "images/disaster_001_pre.png",
  "post": "images/disaster_001_post.png",
  "localization": "labels/disaster_001_localization.png",
  "damage": "labels/disaster_001_damage.png"
}
```

Localization uses `0 = background` and `1 = building`. Damage masks use:

| ID | Damage category |
|---:|---|
| 0 | Background |
| 1 | No damage |
| 2 | Minor damage |
| 3 | Major damage |
| 4 | Destroyed |
| 255 | Unclassified / ignored |

Rasterize xBD polygons onto their corresponding RGB image grids before creating the manifest. Keep unclassified building pixels as buildings in the localization mask and assign `255` in the damage mask.

Manifest fields `localization_ignore_index`, `localization_positive_value`, and `damage_ignore_index` can describe alternative source encodings. The default localization encoding is `0/1`, with `255` ignored. To use a `0/255` localization mask, set `localization_positive_value` to `255` and `localization_ignore_index` to `null`.

## Change captioning

Set `task` to `cc` and provide all reference descriptions for every image pair:

```json
{
  "id": "pair_001",
  "pre": "images/A/001.png",
  "post": "images/B/001.png",
  "captions": [
    "a building appears beside the road",
    "a new building is built near the road",
    "a building has been constructed beside the road",
    "a building appears next to the road",
    "a new building appears near the road"
  ]
}
```

Each reference can be a string or a pre-tokenized list of strings. Strings are split at whitespace; supplied tokens retain their case and punctuation. Training visits each reference, while evaluation visits each image pair once and retains every reference.

The wordmap is built only from training captions. LEVIR-CC uses `min_word_freq = 5`, retaining words with a count strictly greater than five. DUBAI-CC uses `min_word_freq = 0`, retaining every word observed in the training set. Both use `max_words = 50` and four special tokens: `<pad>`, `<unk>`, `<start>`, and `<end>`.

The caption loader resizes RGB inputs to `256 × 256` with Pillow bilinear interpolation and applies ImageNet mean/std normalization. DUBAI-CC's source `50 × 50` images can be referenced directly in the manifest.

## Dataset sources

| Dataset | Original project or release |
|---|---|
| LEVIR-CD | [Dataset project](https://justchenhao.github.io/LEVIR/) |
| WHU-CD | [WHU building dataset](https://gpcv.whu.edu.cn/data/building_dataset.html) |
| SYSU-CD | [SYSU-CD repository](https://github.com/liumency/SYSU-CD) |
| S2Looking | [S2Looking repository](https://github.com/S2Looking/Dataset) |
| SECOND | [SECOND dataset project](https://captain-whu.github.io/SCD/) |
| Landsat-SCD | [Original dataset release](https://doi.org/10.6084/m9.figshare.19946135.v1) |
| JL1H | [CdSC repository](https://github.com/weiAI1996/CdSC) |
| xBD | [xView2 baseline and data instructions](https://github.com/DIUx-xView/xView2_baseline) |
| LEVIR-CC | [LEVIR-CC dataset repository](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) |
| DUBAI-CC | [Original dataset provider](https://disi.unitn.it/~melgani/datasets.html) |

Use the original dataset splits when available. A validation split can be created from training pairs when the source provides only training and test sets.
