"""Compact models for binary/semantic change, damage, and caption prediction."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ._vendor.x3d import create_x3d
from ._vendor.change_decoder import ChangeDecoder

TASKS = ("bcd", "scd", "bda", "cc")
VARIANTS = ("control", "conditional", "conditional_multiplicative", "event_multiplicative")


def state_hash(model):
    """Stable tensor fingerprint, independent of checkpoint serialization."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initialize_decoder(module):
    for child in module.children():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(child.weight, mode="fan_in", nonlinearity="relu")
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        else:
            initialize_decoder(child)


def spatial_sincos(grid, channels):
    quarter = channels // 4
    frequency = torch.exp(-math.log(10000.0) * torch.arange(quarter).float() / quarter)
    y, x = torch.meshgrid(torch.arange(grid).float(), torch.arange(grid).float(), indexing="ij")
    x, y = x.reshape(-1, 1) * frequency, y.reshape(-1, 1) * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)[None]


class StageRoleUpdate(nn.Module):
    """Endpoint-guided state update and state-guided difference modulation."""

    def __init__(self, channels, bottleneck_ratio=4, state_update=True,
                 modulation=True, event_gate_init=0.01):
        super().__init__()
        self.state_update, self.modulation = state_update, modulation
        if state_update:
            self.A = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.ReLU())
            self.alpha_s = nn.Parameter(torch.zeros(()))
        if modulation:
            hidden = max(1, channels // bottleneck_ratio)
            self.B = nn.Sequential(nn.GroupNorm(4, 4 * channels, affine=False),
                                   nn.Conv2d(4 * channels, hidden, 1, bias=False), nn.ReLU(),
                                   nn.Conv2d(hidden, channels, 1, bias=False))
            self.alpha_c = nn.Parameter(torch.tensor(float(event_gate_init)))

    @staticmethod
    def add_residual(value, scale, update):
        with torch.autocast(device_type=value.device.type, enabled=False):
            return (value.float() + scale.float() * update.float()).to(value.dtype)

    def forward(self, value, native_change, variant):
        first, pre, event, post, last = value.unbind(dim=2)
        delta = last - first
        if self.state_update:
            a, b = (pre, post) if variant == "control" else (first, last)
            pre = self.add_residual(pre, self.alpha_s, self.A(a))
            post = self.add_residual(post, self.alpha_s, self.A(b))
        change = native_change
        if self.modulation:
            if variant in ("conditional", "conditional_multiplicative"):
                condition = (pre, post, delta.abs(), delta)
            elif variant == "event_multiplicative":
                condition = (event, event, delta.abs(), delta)
            else:
                condition = (event, event, delta.abs(), delta.abs())
            combined = torch.cat(condition, dim=1)
            if variant in ("conditional_multiplicative", "event_multiplicative"):
                normalized = self.B[0](combined)
                split = normalized.shape[1] // 2
                weight = self.B[1].weight
                state = F.conv2d(normalized[:, :split], weight[:, :split])
                difference = F.conv2d(normalized[:, split:], weight[:, split:])
                update = self.B[3](self.B[2](difference) * (1 + state.tanh()))
            else:
                update = self.B(combined)
            change = self.add_residual(change, self.alpha_c, update)
        result = value.clone()
        result[:, :, 1], result[:, :, 2], result[:, :, 3] = pre, change, post
        return result


class ChangeModel(nn.Module):
    """Shared five-frame encoder and a task-specific prediction head.

    Dense inputs must have height and width divisible by eight. Prediction
    helpers pad arbitrary input images and crop the output back to native size.
    """

    channels = (24, 24, 48, 96)

    def __init__(self, task="bcd", *, num_classes=7, vocab_size=None, grid_size=16,
                 depth_factor=5.0, bottleneck_ratio=4, variant="conditional_multiplicative",
                 event_gate_init=0.01, state_update=True, modulation=True,
                 modulation_scales=None, readout="role", pretrained_backbone=None,
                 pad_id=0, d_model=128, num_heads=4, feedforward_dim=256,
                 decoder_layers=1, dropout=0.1, max_length=52, memory_grid=8):
        super().__init__()
        if task not in TASKS or variant not in VARIANTS:
            raise ValueError(f"task must be in {TASKS}; variant must be in {VARIANTS}")
        if grid_size < 1 or depth_factor <= 0 or bottleneck_ratio < 1 or num_classes < 2:
            raise ValueError("Grid, depth, bottleneck ratio, and class count must be positive")
        if readout not in ("role", "mean", "max"):
            raise ValueError("readout must be role, mean, or max")
        scales = tuple(range(4)) if modulation_scales is None else tuple(modulation_scales)
        if any(type(s) is not int or s not in range(4) for s in scales):
            raise ValueError("modulation_scales contains zero-based encoder scale indices 0..3")
        self.task, self.variant, self.readout = task, variant, readout
        self.config = dict(task=task, num_classes=num_classes, vocab_size=vocab_size,
                           grid_size=grid_size, depth_factor=depth_factor,
                           bottleneck_ratio=bottleneck_ratio, variant=variant,
                           event_gate_init=event_gate_init, state_update=state_update,
                           modulation=modulation, modulation_scales=list(scales), readout=readout,
                           pad_id=pad_id, d_model=d_model, num_heads=num_heads,
                           feedforward_dim=feedforward_dim, decoder_layers=decoder_layers,
                           dropout=dropout, max_length=max_length, memory_grid=memory_grid)
        backbone = create_x3d(input_clip_length=3, depth_factor=float(depth_factor))
        self.backbone_load_info = None
        if pretrained_backbone is not None:
            self.backbone_load_info = self._load_backbone(backbone, pretrained_backbone)
        self.student = nn.ModuleList(list(backbone.blocks[:4]))
        del backbone
        self.perception_grids = nn.Parameter(torch.randn(1, 3, 3, grid_size, grid_size))
        self.enhance = nn.ModuleList(nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.ReLU())
                                     for c in self.channels)
        # Preserve the original head-construction order for checkpoint compatibility.
        self.heads = nn.ModuleDict({
            "pre": ChangeDecoder(SimpleNamespace(num_class=7), in_dim=list(self.channels)),
            "post": ChangeDecoder(SimpleNamespace(num_class=7), in_dim=list(self.channels)),
            "change": ChangeDecoder(SimpleNamespace(num_class=1), in_dim=list(self.channels), has_sigmoid=True),
        })
        for decoder in self.heads.values():
            initialize_decoder(decoder)
        with torch.random.fork_rng(devices=[]):
            self.role_updates = nn.ModuleList(
                StageRoleUpdate(c, bottleneck_ratio, state_update,
                                modulation and index in scales, event_gate_init)
                for index, c in enumerate(self.channels))
        if task == "scd" and num_classes != 7:
            for role in ("pre", "post"):
                self.heads[role].up_c1[0] = nn.Conv2d(24, num_classes, 3, padding=1, bias=False)
                initialize_decoder(self.heads[role].up_c1)
        elif task == "bcd":
            del self.heads["pre"], self.heads["post"]
            self.heads["change"].has_sigmoid = False
            with torch.random.fork_rng(devices=[]):
                self.bcd_fusion = nn.Conv2d(288, 96, 1, bias=False)
                with torch.no_grad():
                    self.bcd_fusion.weight.zero_()
                    self.bcd_fusion.weight[:, 96:192, 0, 0].copy_(torch.eye(96))
        elif task == "bda":
            self.heads = nn.ModuleDict()
            with torch.random.fork_rng(devices=[]):
                self.heads["localization"] = ChangeDecoder(SimpleNamespace(num_class=1), in_dim=list(self.channels))
                self.heads["damage"] = ChangeDecoder(SimpleNamespace(num_class=5), in_dim=list(self.channels))
                for decoder in self.heads.values():
                    initialize_decoder(decoder)
                self.damage_fusion = nn.ModuleList(nn.Conv2d(3*c, c, 1, bias=False) for c in self.channels)
                with torch.no_grad():
                    for layer, c in zip(self.damage_fusion, self.channels):
                        layer.weight.zero_()
                        for slot, scale in enumerate((0.25, 0.5, 0.25)):
                            layer.weight[:, slot*c:(slot+1)*c, 0, 0].copy_(scale*torch.eye(c))
        elif task == "cc":
            if not isinstance(vocab_size, int) or vocab_size < 4:
                raise ValueError("Captioning requires a vocabulary of at least four tokens")
            if not 0 <= pad_id < vocab_size or d_model % num_heads or d_model % 4:
                raise ValueError("Invalid caption embedding or attention dimensions")
            self.heads = nn.ModuleDict()
            self.vocab_size, self.pad_id = vocab_size, pad_id
            self.d_model, self.max_length, self.memory_grid = d_model, max_length, memory_grid
            with torch.random.fork_rng(devices=[]):
                self.memory_projection = nn.Conv2d(288, d_model, 1, bias=False)
                self.memory_norm = nn.LayerNorm(d_model)
                self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
                self.caption_position = nn.Embedding(max_length, d_model)
                self.caption_dropout = nn.Dropout(dropout)
                layer = nn.TransformerDecoderLayer(d_model, num_heads, feedforward_dim, dropout,
                                                   activation="relu", batch_first=True)
                self.decoder = nn.TransformerDecoder(layer, decoder_layers, norm=nn.LayerNorm(d_model))
                self.output_bias = nn.Parameter(torch.zeros(vocab_size))
                nn.init.normal_(self.token_embedding.weight, std=d_model**-0.5)
                with torch.no_grad():
                    self.token_embedding.weight[pad_id].zero_()
                nn.init.normal_(self.caption_position.weight, std=0.02)
            self.register_buffer("spatial_position", spatial_sincos(memory_grid, d_model), persistent=False)

    @staticmethod
    def _load_backbone(backbone, path):
        checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        current = backbone.state_dict()
        matching = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        if not matching:
            raise ValueError("No matching X3D backbone tensors found")
        mismatch = backbone.load_state_dict(matching, strict=False)
        return dict(loaded=len(matching), missing=list(mismatch.missing_keys),
                    unused=[k for k in state if k not in matching])

    def parameter_report(self):
        total = sum(p.numel() for p in self.parameters())
        return dict(task=self.task, total=total,
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    stage_blocks=[len(block.res_blocks) for block in list(self.student)[1:]],
                    grid_parameters=self.perception_grids.numel(),
                    modules={name: sum(p.numel() for p in module.parameters())
                             for name, module in self.named_children()})

    def encode(self, pre, post):
        if pre.ndim != 4 or pre.shape[1] != 3 or pre.shape != post.shape:
            raise ValueError("Expected matching Bx3xHxW input tensors")
        if min(pre.shape[-2:]) < 16 or any(v % 8 for v in pre.shape[-2:]):
            raise ValueError("Input height and width must be >=16 and divisible by eight")
        grids = self.perception_grids[0].permute(1, 0, 2, 3)
        frames = F.interpolate(grids, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        frames = frames.permute(1, 0, 2, 3)[None].expand(pre.shape[0], -1, -1, -1, -1)
        value = torch.cat((pre.unsqueeze(2), frames, post.unsqueeze(2)), dim=2)
        roles = ([], [], [])
        for block, enhance, update in zip(self.student, self.enhance, self.role_updates):
            value = block(value)
            native = value[:, :, 2] + enhance((value[:, :, 0]-value[:, :, 4]).abs())
            value = update(value, native, self.variant)
            maps = [value[:, :, i] for i in (1, 2, 3)]
            if self.readout != "role":
                stack = torch.stack(maps)
                merged = stack.mean(0) if self.readout == "mean" else stack.max(0).values
                maps = [merged] * 3
            for output, feature in zip(roles, maps):
                output.append(feature)
        return roles

    def forward(self, pre, post, input_tokens=None):
        if self.task == "cc":
            if input_tokens is None:
                raise ValueError("Caption training requires input_tokens; use generate() for prediction")
            return {"caption_logits": self.decode_tokens(self.encode_memory(pre, post), input_tokens)}
        state_pre, change, state_post = self.encode(pre, post)
        if self.task == "scd":
            return {"pred_a": self.heads["pre"](state_pre), "pred_b": self.heads["post"](state_post),
                    "pred_cd": self.heads["change"](change)}
        if self.task == "bcd":
            change[-1] = self.bcd_fusion(torch.cat((state_pre[-1], change[-1], state_post[-1]), dim=1))
            logits = self.heads["change"](change)
            return {"change_logits": logits, "pred_cd": logits}
        fused = [layer(torch.cat(parts, dim=1))
                 for layer, parts in zip(self.damage_fusion, zip(state_pre, change, state_post))]
        return {"localization_logits": self.heads["localization"](state_pre),
                "damage_logits": self.heads["damage"](fused)}

    def encode_memory(self, pre, post):
        roles = self.encode(pre, post)
        feature = F.adaptive_avg_pool2d(torch.cat([r[-1] for r in roles], dim=1),
                                      (self.memory_grid, self.memory_grid))
        memory = self.memory_projection(feature).flatten(2).transpose(1, 2)
        return self.memory_norm(memory + self.spatial_position.to(dtype=memory.dtype))

    def decode_tokens(self, memory, input_tokens):
        if input_tokens.ndim != 2 or input_tokens.dtype != torch.long:
            raise ValueError("Caption input_tokens must be an int64 BxL tensor")
        length = input_tokens.shape[1]
        if not 1 <= length <= self.max_length or input_tokens.shape[0] != memory.shape[0]:
            raise ValueError("Caption prefix length or batch size does not match")
        position = torch.arange(length, device=input_tokens.device)
        target = self.token_embedding(input_tokens) * math.sqrt(self.d_model)
        target = self.caption_dropout(target + self.caption_position(position)[None])
        causal = torch.ones(length, length, dtype=torch.bool, device=input_tokens.device).triu(1)
        result = self.decoder(target, memory, tgt_mask=causal,
                              tgt_key_padding_mask=input_tokens.eq(self.pad_id))
        return F.linear(result, self.token_embedding.weight, self.output_bias)

    @torch.no_grad()
    def generate(self, x1, x2, bos_id, eos_id, *, max_length=None, beam_size=1):
        """Return Bxmax_length token IDs including BOS; pad after first EOS.

        Beam search follows the official cumulative-log-probability scoring
        without length normalization. Finished beams leave the active set.
        The implementation batches all image/beam decoder calls at each step.
        """
        if self.training:
            raise ValueError('Caption generation requires eval()')
        max_length = self.max_length if max_length is None else max_length
        if type(max_length) is not int or not 2 <= max_length <= self.max_length:
            raise ValueError('Invalid generation length')
        if type(beam_size) is not int or not 1 <= beam_size <= self.vocab_size-2:
            raise ValueError('Invalid beam size')
        if not (0 <= bos_id < self.vocab_size and 0 <= eos_id < self.vocab_size
                and len({self.pad_id, bos_id, eos_id}) == 3):
            raise ValueError('BOS, EOS and PAD must be distinct vocabulary IDs')
        memory = self.encode_memory(x1, x2)
        batch = memory.shape[0]
        device = memory.device
        sequences = torch.full((batch, beam_size, 1), bos_id, dtype=torch.long, device=device)
        scores = torch.full((batch, beam_size), float('-inf'), device=device)
        scores[:, 0] = 0.
        active = torch.ones(batch, dtype=torch.bool, device=device)
        completed_scores = torch.full((batch,), float('-inf'), device=device)
        completed_sequences = torch.full((batch, max_length), self.pad_id,
                                         dtype=torch.long, device=device)
        # Each sample keeps K slots; finished hypotheses consume its remaining
        # budget exactly as in the official shrinking-beam implementation.
        slots = torch.full((batch,), beam_size, dtype=torch.long, device=device)
        rank = torch.arange(beam_size, device=device)[None]
        flat_memory = memory[:, None].expand(-1, beam_size, -1, -1).reshape(
            batch*beam_size, memory.shape[1], memory.shape[2])
        for _ in range(1, max_length):
            logp = self.decode_tokens(flat_memory, sequences.flatten(0, 1))[:, -1].float().log_softmax(-1)
            logp[:, self.pad_id] = float('-inf')
            logp[:, bos_id] = float('-inf')
            candidate = logp.reshape(batch, beam_size, self.vocab_size) + scores[..., None]
            values, indices = candidate.flatten(1).topk(beam_size, dim=1)
            parents, words = indices//self.vocab_size, indices % self.vocab_size
            selected = torch.cat((sequences.gather(1, parents[..., None].expand(
                -1, -1, sequences.shape[-1])), words[..., None]), dim=-1)
            valid = (rank < slots[:, None]) & torch.isfinite(values) & active[:, None]
            finished = valid & words.eq(eos_id)
            finish_scores, finish_indices = values.masked_fill(~finished, float('-inf')).max(dim=1)
            improved = finish_scores > completed_scores
            best_finished = selected[torch.arange(batch, device=device), finish_indices]
            padded = F.pad(best_finished, (0, max_length-best_finished.shape[1]), value=self.pad_id)
            completed_sequences = torch.where(improved[:, None], padded, completed_sequences)
            completed_scores = torch.maximum(completed_scores, finish_scores)
            slots = slots-finished.sum(dim=1)
            # Pack live hypotheses first, preserving candidate score order.
            live = valid & ~finished
            order = torch.where(live, rank, rank+beam_size).argsort(dim=1)
            scores = values.masked_fill(~live, float('-inf')).gather(1, order)
            sequences = selected.gather(1, order[..., None].expand(-1, -1, selected.shape[-1]))
            active = (slots > 0) & torch.isfinite(scores).any(dim=1)
            if not bool(active.any()):
                break
        # Match conventional EOS-completed preference; use unfinished beams
        # only for samples without a completed caption at the length limit.
        has_completed = torch.isfinite(completed_scores)
        best_active_scores, best_active_indices = scores.max(dim=1)
        if not bool((has_completed | torch.isfinite(best_active_scores)).all()):
            raise RuntimeError('Beam search produced no finite hypothesis')
        unfinished = sequences[torch.arange(batch, device=device), best_active_indices]
        unfinished = F.pad(unfinished, (0, max_length-unfinished.shape[1]), value=self.pad_id)
        return torch.where(has_completed[:, None], completed_sequences, unfinished)


def build_model(config=None, **kwargs):
    """Construct a model from keyword arguments or the model section of a config."""
    values = dict(config or {})
    if "model" in values:
        values = dict(values["model"], task=values["task"])
    values.update(kwargs)
    return ChangeModel(**values)
