"""Task objectives for binary/semantic change, damage assessment, and captions."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Loss values and boolean validity mask must share their shape")
    return values[valid].mean() if bool(valid.any()) else values.sum()*0.


def _valid(batch, reference):
    value = batch.get("valid_mask")
    return torch.ones_like(reference, dtype=torch.bool) if value is None else value.bool()


def _binary_loss(logits, target, valid, sample_dice=False):
    if target.ndim == 3:
        target = target[:, None]
    if valid.ndim == 3:
        valid = valid[:, None]
    if target.shape != logits.shape or valid.shape != logits.shape:
        raise ValueError("Binary logits, targets, and validity mask must have matching shapes")
    target = target.float()
    binary = masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid)
    probability = logits.sigmoid()
    if sample_dice:
        p, y = probability*valid, target*valid
        dice = 1-(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)
        dice = masked_mean(dice, valid.flatten(1).any(1))
    else:
        p, y = probability[valid], target[valid]
        dice = 1-(2*(p*y).sum()+1e-5)/(p.sum()+y.sum()+1e-5)
    return binary, dice


def task_loss(task: str, out, batch: dict, *, pseudo_weight: float = 1.) -> tuple[torch.Tensor, dict]:
    """Return a differentiable scalar and its component scalars.

    Cached SCD pseudo labels are optional, agreed nonzero semantic labels at
    original zero/zero pixels. Their CE is added without changing supervised
    targets, binary/Dice targets, or the similarity objective.
    """
    if not math.isfinite(float(pseudo_weight)) or pseudo_weight < 0:
        raise ValueError("pseudo_weight must be finite and nonnegative")
    probe = out if isinstance(out, torch.Tensor) else next(v for v in out.values() if isinstance(v, torch.Tensor))
    with torch.autocast(device_type=probe.device.type, enabled=False):
        if task == "bcd":
            logits = out["change_logits"].float()
            label = batch["label"]
            binary, dice = _binary_loss(logits, label, _valid(batch, label[:,0] if label.ndim == 4 else label), True)
            parts = {"binary": binary, "dice": dice}
        elif task == "scd":
            a, b, probability = [out[k].float() for k in ("pred_a", "pred_b", "pred_cd")]
            ya, yb = batch["label_pre"].long(), batch["label_post"].long()
            valid = _valid(batch, ya)
            if not torch.equal((ya > 0)[valid], (yb > 0)[valid]):
                raise ValueError("SCD endpoint targets disagree on the changed region")
            ya, yb = [torch.where(valid, y, torch.zeros_like(y)) for y in (ya, yb)]
            changed = ya > 0
            semantic = .5*sum(masked_mean(F.cross_entropy(logits, target, reduction="none"), valid & (target > 0))
                              for logits, target in ((a,ya),(b,yb)))
            if probability.ndim == 4:
                if probability.shape[1] != 1:
                    raise ValueError("The SCD change branch must have one output channel")
                probability = probability[:,0]
            if probability.shape != ya.shape:
                raise ValueError("SCD change probabilities must match the target shape")
            binary = masked_mean(F.binary_cross_entropy(probability, changed.float(), reduction="none"), valid)
            p, y = probability[valid], changed.float()[valid]
            dice = 1-(2*(p*y).sum()+1e-5)/(p.sum()+y.sum()+1e-5)
            classes = a.shape[1]-1
            pa, pb = [F.softmax(x[:,1:], dim=1).permute(0,2,3,1).reshape(-1,classes) for x in (a,b)]
            sign = torch.where(changed, -1., 1.).flatten()
            similarity = F.cosine_embedding_loss(pa, pb, sign, margin=0., reduction="none")
            similarity = masked_mean(similarity.masked_fill(((ya == yb) & changed).flatten(), 0.), valid.flatten())
            parts = {"semantic": semantic, "binary": binary, "dice": dice, "similarity": similarity}
            if "pseudo_pre" in batch or "pseudo_post" in batch:
                if "pseudo_pre" not in batch or "pseudo_post" not in batch:
                    raise ValueError("Cached pseudo labels must provide both endpoints")
                pya, pyb = batch["pseudo_pre"].long(), batch["pseudo_post"].long()
                accepted = valid & (ya == 0) & (yb == 0) & (pya > 0) & (pya == pyb)
                accepted &= (pya < a.shape[1]) & (pyb < b.shape[1])
                safe_a = torch.where(accepted, pya, torch.zeros_like(pya))
                safe_b = torch.where(accepted, pyb, torch.zeros_like(pyb))
                pseudo = .5*sum(masked_mean(F.cross_entropy(logits, target, reduction="none"), accepted)
                                for logits,target in ((a,safe_a),(b,safe_b)))
                parts["pseudo"] = pseudo*float(pseudo_weight)
        elif task == "bda":
            logits = out.get("localization_logits", out.get("loc_logits"))
            if logits is None:
                raise ValueError("BDA output needs localization_logits")
            logits, damage_logits = logits.float(), out["damage_logits"].float()
            loc, damage = batch["localization"].long(), batch["damage"].long()
            valid = _valid(batch, loc) & (loc != 255)
            binary, dice = _binary_loss(logits, loc.eq(1).float(), valid)
            target = torch.where(loc == 1, damage, torch.zeros_like(damage))
            target = target.masked_fill(~valid, 255)
            classification = (F.cross_entropy(damage_logits, target, ignore_index=255)
                              if bool((target != 255).any()) else damage_logits.sum()*0.)
            parts = {"binary": binary, "dice": dice, "damage": classification}
        elif task == "cc":
            logits = out.float() if isinstance(out, torch.Tensor) else out["caption_logits"].float()
            target = batch["tokens"][:,1:]
            if logits.shape[:2] != target.shape:
                raise ValueError("Caption logits must align with tokens[:, 1:]; feed tokens[:, :-1] to the decoder")
            ce = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), target.reshape(-1), ignore_index=0)
            parts = {"caption_ce": ce}
        else:
            raise ValueError(f"Unknown task {task!r}")
        total = sum(parts.values())
    return total, parts
