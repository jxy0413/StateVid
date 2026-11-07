# Training, evaluation, and prediction

## Command-line entry points

| Command | Purpose | Module equivalent |
|---|---|---|
| `statevid-train` | Train a dataset-specific model | `python -m statevid.cli train` |
| `statevid-eval` | Evaluate a checkpoint on a manifest split | `python -m statevid.cli evaluate` |
| `statevid-predict` | Export predictions from one pair or a manifest | `python -m statevid.cli predict` |
| `statevid-inspect` | Inspect model settings and parameter counts | `python -m statevid.cli inspect` |

Use `--help` with any command to see its options.

## Train from a configuration

```bash
statevid-train --config configs/bcd_levir_cd.json \
  --output runs/levir_cd --device cuda
```

Choose a physical GPU with `CUDA_VISIBLE_DEVICES`, for example:

```bash
CUDA_VISIBLE_DEVICES=1 statevid-train \
  --config configs/scd_second.json --output runs/second --device cuda
```

Use separate output directories for independent tasks or variants. Configuration and training outputs are saved under the selected run directory. The best checkpoint is selected using the validation split.

Resume an interrupted run from its saved training checkpoint:

```bash
statevid-train --config configs/bcd_levir_cd.json \
  --output runs/levir_cd --resume runs/levir_cd/last.pt --device cuda
```

### Initialization

The supplied configurations set `model.pretrained_backbone` to `null` and train the model from scratch.

To use generic X3D-L initialization, download the original backbone with the optional utility:

```bash
python scripts/download_backbone.py --output weights/X3D_L.pyth
```

Then set:

```json
{
  "model": {
    "pretrained_backbone": "weights/X3D_L.pyth"
  }
}
```

The utility downloads the [original PyTorchVideo X3D-L file](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth) and checks its SHA-256:

```text
19c99d8ab8f2aceab4f8a59a85f6dbc3c7f3a262a798ca95b906709ed100866b
```

The task-specific perception grids, modulation modules, and decoders are trained together with the retained encoder.

## Evaluate a dataset split

```bash
statevid-eval --checkpoint runs/levir_cd/best.pt \
  --manifest data/levir_cd/manifest.json \
  --split test --output runs/levir_cd/test_metrics.json --device cuda
```

The evaluator reconstructs the model and caption vocabulary from the checkpoint. Metrics are computed over the supplied split. Dense metrics use valid pixels; ignored and padded pixels do not contribute.

Pixel metrics are returned as fractions. Caption metrics use the selected backend's native scale.

For large dense scenes, add `--tile-size 256` to evaluate non-overlapping native tiles. The same setting is available as `train.eval_tile_size` for validation during training.

| Task | Metrics |
|---|---|
| BCD | Changed-class F1 and IoU, precision, recall |
| SCD | Binary mIoU, separated Kappa (SeK), semantic change F1 |
| BDA | Localization F1, per-category damage F1, harmonic damage F1, weighted overall F1 |
| CC | BLEU-4 and ROUGE-L; full caption extra adds METEOR and CIDEr |

BDA damage classification is evaluated at ground-truth building pixels with known damage labels. Its overall score is `0.3 × localization F1 + 0.7 × damage classification F1`. The damage classification F1 is the harmonic mean of the four building-damage class F1 values.

## Export one prediction

```bash
statevid-predict --checkpoint runs/levir_cd/best.pt \
  --pre path/to/before.png --post path/to/after.png \
  --output predictions/example --device cuda
```

Supply an RGB image pair with matching spatial dimensions. The command exports the task's label maps or generated caption. Dense images retain their field of view; caption images use the checkpoint's image-size and normalization settings.

| Task | Files written per pair |
|---|---|
| BCD | `change_mask.png` |
| SCD | `pre_labels.png`, `post_labels.png`, `change_mask.png` |
| BDA | `localization_mask.png`, `damage_labels.png`, `damage_map.png` |
| CC | `caption.json`, `caption.txt` |

Binary PNG masks use `0/255`. Semantic and damage PNGs contain class IDs; `damage_map.png` applies the predicted localization mask to the damage labels.

## Export a manifest subset

```bash
statevid-predict --checkpoint runs/second/best.pt \
  --manifest data/second/manifest.json --split test --limit 3 \
  --output predictions/second --device cuda
```

The prediction command can also read manifests without labels. In that case, include each pair's `id`, `pre`, and `post` fields and the manifest's task information.

## Caption generation and full metrics

The caption decoder uses autoregressive beam search. The examples set beam size three and retain the vocabulary constructed from training captions. Changing a dataset's vocabulary requires constructing a matching caption model.

For complete caption metrics:

```bash
python -m pip install -e '.[caption]'
```

Install a Java runtime to run METEOR. Set `caption.metric_backend` to `coco` and `train.selection_metric` to `CIDEr` when choosing the best checkpoint with that metric.

An existing caption checkpoint can be evaluated with the complete backend directly:

```bash
statevid-eval --checkpoint runs/levir_cc/best.pt \
  --manifest data/levir_cc/manifest.json --split test \
  --caption-backend coco --include-meteor \
  --output runs/levir_cc/full_caption_metrics.json --device cuda
```

To include METEOR during validation, also set `caption.include_meteor` to `true`.

## Model inspection

```bash
statevid-inspect --task bcd --input-size 256
```

The [model architecture guide](model.md) gives encoder scales, update equations, task readouts, and parameter formulas.
