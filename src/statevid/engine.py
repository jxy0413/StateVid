"""Training, validation, evaluation, and portable model serialization."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ChangeDataset, build_wordmap, collate_batch, load_manifest
from .losses import task_loss
from .metrics import create_metric, update_metrics
from .models import build_model
from .pseudo import pseudo_loss_weight


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    root = next((p for p in path.parents if (p / "pyproject.toml").is_file()), Path.cwd())
    for key in ("manifest",):
        value = Path(config[key]).expanduser()
        config[key] = str(value if value.is_absolute() else root / value)
    pretrained = config.get("model", {}).get("pretrained_backbone")
    if pretrained:
        value = Path(pretrained).expanduser()
        config["model"]["pretrained_backbone"] = str(value if value.is_absolute() else root / value)
    return config


def select_device(name=None):
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _loader(dataset, batch_size, workers, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate_batch, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0, generator=generator)


@torch.inference_mode()
def evaluate_model(model, dataset, *, device, batch_size=2, workers=0, wordmap=None,
                   beam_size=3, caption_backend="builtin", include_meteor=False):
    model.eval()
    meter = create_metric(model.task, model.config["num_classes"])
    loss_sum = 0.0
    examples = 0
    for batch in _loader(dataset, batch_size, workers):
        batch = move_batch(batch, device)
        if model.task == "cc":
            output = model(batch["pre"], batch["post"], batch["tokens"][:, :-1])
            loss, _ = task_loss(model.task, output, batch)
            output["generated_tokens"] = model.generate(batch["pre"], batch["post"],
                                                        wordmap["<start>"], wordmap["<end>"],
                                                        beam_size=beam_size)
        else:
            output = model(batch["pre"], batch["post"])
            loss, _ = task_loss(model.task, output, batch, pseudo_weight=0)
        update_metrics(model.task, meter, output, batch, wordmap=wordmap)
        count = batch["pre"].shape[0]
        loss_sum += float(loss) * count
        examples += count
    if not examples:
        raise ValueError("Evaluation split is empty")
    values = (meter.compute(backend=caption_backend, include_meteor=include_meteor)
              if model.task == "cc" else meter.compute())
    return dict(values, loss=loss_sum / examples, samples=examples)


def save_checkpoint(path, model, optimizer, config, wordmap, step, epoch, best_score):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    state = dict(model={key: value.detach().cpu() for key, value in model.state_dict().items()},
                 model_config=model.config, optimizer=optimizer.state_dict(), config=config,
                 wordmap=wordmap, step=step, epoch=epoch, best_score=best_score,
                 torch_rng_state=torch.get_rng_state(), python_rng_state=random.getstate())
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint written by this package, with tensor-only unpickling."""
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if "model_config" not in saved or "model" not in saved:
        raise ValueError("Expected a StateVid checkpoint containing model_config and model")
    model = build_model(saved["model_config"])
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device), saved


def train(config, output, *, device=None, max_steps=None, resume=None):
    config = dict(config)
    settings = dict(config.get("train", {}))
    caption = dict(config.get("caption", {}))
    task = config["task"]
    seed = int(settings.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "last.pt").exists():
        raise FileExistsError("Output already contains a run; use --resume or choose another directory")
    manifest = load_manifest(config["manifest"], task)
    wordmap = None
    model_config = dict(config.get("model", {}), task=task)
    if task == "scd":
        declared = model_config.get("num_classes", manifest["num_classes"])
        if declared != manifest["num_classes"]:
            raise ValueError("Configured num_classes disagrees with the manifest")
        model_config["num_classes"] = declared
    if task == "cc":
        wordmap, metadata = build_wordmap(config["manifest"],
                                         min_word_freq=caption.get("min_word_freq", 5),
                                         max_words=caption.get("max_words", 50))
        model_config.update(vocab_size=len(wordmap), pad_id=wordmap["<pad>"],
                            max_length=caption.get("max_words", 50) + 2)
        write_json(output / "wordmap.json", wordmap)
        write_json(output / "vocabulary.json", metadata)
    model = build_model(model_config).to(device)
    train_data = ChangeDataset(config["manifest"], "train", crop_size=settings.get("crop_size", 256),
                               wordmap=wordmap, max_words=caption.get("max_words", 50),
                               cc_image_size=int(caption.get("image_size", 256)))
    val_data = ChangeDataset(config["manifest"], "val", augment=False, wordmap=wordmap,
                             max_words=caption.get("max_words", 50),
                             tile_size=settings.get("eval_tile_size"),
                             cc_image_size=int(caption.get("image_size", 256)))
    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError("Training and validation splits must be nonempty")
    pseudo_max = float(settings.get("pseudo_weight", 0))
    if task == "scd" and pseudo_max > 0:
        if any(not row.get("pseudo_pre") or not row.get("pseudo_post")
               for row in manifest["splits"]["train"]):
            raise ValueError("Positive pseudo_weight requires cached pseudo_pre/post paths in each training record")
    generator = torch.Generator().manual_seed(seed)
    workers = int(settings.get("workers", 4))
    loader = _loader(train_data, int(settings.get("batch_size", 8)), workers, True, generator)
    accumulation = int(settings.get("accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("accumulation_steps must be positive")
    epochs = int(settings.get("epochs", 100))
    total_steps = max_steps or settings.get("max_steps") or math.ceil(len(loader) / accumulation) * epochs
    if total_steps < 1:
        raise ValueError("Training budget must be positive")
    base_lr = float(settings.get("lr", 2e-4))
    regular, modulation = [], []
    for name, parameter in model.named_parameters():
        is_modulation = name.startswith("role_updates.") and (".B." in name or name.endswith(".alpha_c"))
        (modulation if is_modulation else regular).append(parameter)
    optimizer = torch.optim.Adam([
        {"params": regular, "weight_decay": float(settings.get("weight_decay", 1e-4))},
        {"params": modulation, "weight_decay": 0.0},
    ], lr=base_lr, betas=tuple(settings.get("betas", [0.9, 0.99])))
    use_amp = bool(settings.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    step, start_epoch, best = 0, 0, float("-inf")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["model_config"] != model.config or saved.get("wordmap") != wordmap:
            raise ValueError("Resume configuration or vocabulary differs from the saved run")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, start_epoch, best = saved["step"], saved["epoch"], saved["best_score"]
        torch.set_rng_state(saved["torch_rng_state"])
        random.setstate(saved["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        if total_steps <= step:
            raise ValueError("The resumed training budget must exceed the saved step")
    selection = settings.get("selection_metric", {"bcd": "F1", "scd": "SeK", "bda": "F1_overall", "cc": "ROUGE_L"}[task])
    backend = caption.get("metric_backend", "builtin")
    if task == "cc" and selection in ("CIDEr", "METEOR") and backend != "coco":
        raise ValueError("CIDEr/METEOR selection requires caption.metric_backend='coco'")
    if task == "cc" and backend == "coco":
        import importlib.util
        if importlib.util.find_spec("pycocoevalcap") is None:
            raise ImportError("Install the caption extra: pip install '.[caption]'")
    eval_every = max(1, int(settings.get("eval_every", 1000)))
    warmup = int(settings.get("warmup_steps", min(500, total_steps // 10)))
    write_json(output / "config.json", config)
    write_json(output / "model.json", model.parameter_report())
    started = time.monotonic()
    epoch = start_epoch
    last_metrics = None
    while step < total_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            window = min(accumulation, len(loader) - (index // accumulation) * accumulation)
            ratio = ((step + 1) / warmup if step < warmup else
                     0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step-warmup) / max(1, total_steps-warmup))))
            for group in optimizer.param_groups:
                group["lr"] = base_lr * ratio
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                result = (model(batch["pre"], batch["post"], batch["tokens"][:, :-1])
                          if task == "cc" else model(batch["pre"], batch["post"]))
            pseudo_weight = (pseudo_loss_weight(epoch,
                             warmup_epochs=int(settings.get("pseudo_warmup_epochs", 10)),
                             ramp_epochs=int(settings.get("pseudo_ramp_epochs", 5)),
                             max_weight=pseudo_max) if settings.get("pseudo_schedule", True) else pseudo_max)
            loss, parts = task_loss(task, result, batch, pseudo_weight=pseudo_weight)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss / window).backward()
            if (index + 1) % accumulation != 0 and index + 1 != len(loader):
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("grad_clip", 5)), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % int(settings.get("log_every", 25)) == 0:
                event = dict(step=step, total_steps=total_steps, epoch=epoch, loss=float(loss.detach()),
                             lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic()-started,
                             pseudo_weight=pseudo_weight,
                             **{key: float(value.detach()) for key, value in parts.items() if key != "loss"})
                print(json.dumps(event), flush=True)
                with (output / "train.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
            if step % eval_every == 0 or step == total_steps:
                last_metrics = evaluate_model(model, val_data, device=device, workers=workers,
                                               batch_size=int(settings.get("eval_batch_size", 2)),
                                               wordmap=wordmap, beam_size=int(caption.get("beam_size", 3)),
                                               caption_backend=backend,
                                               include_meteor=bool(caption.get("include_meteor", False)))
                score = float(last_metrics[selection])
                if not math.isfinite(score):
                    raise FloatingPointError(f"Nonfinite validation selection metric: {selection}")
                if score > best:
                    best = score
                    save_checkpoint(output / "best.pt", model, optimizer, config, wordmap, step, epoch, best)
                save_checkpoint(output / "last.pt", model, optimizer, config, wordmap, step, epoch, best)
                write_json(output / "validation.json", dict(step=step, **last_metrics))
                print(json.dumps(dict(validation=True, step=step, **last_metrics)), flush=True)
                model.train()
            if step >= total_steps:
                break
        epoch += 1
    status = dict(state="completed", step=step, best_metric=selection, best_score=best,
                  elapsed_seconds=time.monotonic()-started)
    write_json(output / "status.json", status)
    return status
