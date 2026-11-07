"""Teacher-independent classwise pseudo-label screening for SCD training."""
from __future__ import annotations

import math
import numpy as np
import torch


def pseudo_loss_weight(epoch: int, warmup_epochs: int = 10, ramp_epochs: int = 5,
                       max_weight: float = .2) -> float:
    """Zero-based warm-up followed by a linear ramp to the maximum weight."""
    if type(epoch) is not int or epoch < 0 or type(warmup_epochs) is not int or warmup_epochs < 0:
        raise ValueError("epoch and warmup_epochs must be nonnegative integers")
    if type(ramp_epochs) is not int or ramp_epochs < 1 or not math.isfinite(max_weight) or max_weight < 0:
        raise ValueError("ramp_epochs must be positive and max_weight finite/nonnegative")
    active_epoch = epoch-warmup_epochs+1
    return 0. if active_epoch <= 0 else float(max_weight)*min(active_epoch/ramp_epochs,1.)


class ClasswisePseudoLabeler:
    """Cumulative, count-weighted 60th-percentile confidence thresholds.

    Each class starts with one prior observation at confidence 0.6. The
    current batch percentile is weighted by the number of pixels assigned
    to that class, and the running average is clipped to [0.5, 0.9]. The
    first endpoint updates the shared accumulator before its confidence
    test; the second endpoint then updates the same accumulator.
    """
    def __init__(self, num_classes: int = 7):
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must include unchanged class 0 and at least one semantic class")
        self.num_classes = num_classes
        self.count = np.ones(num_classes,dtype=np.int64)
        self.confidence_sum = np.full(num_classes,.6,dtype=np.float64)

    @property
    def thresholds(self):
        return np.clip(self.confidence_sum/self.count,.5,.9)

    def state_dict(self):
        return {"version":1,"num_classes":self.num_classes,"initial_threshold":.6,
                "percentile":60.,"clip":[.5,.9],"count":self.count.tolist(),
                "sum":self.confidence_sum.tolist()}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError("Pseudo-label threshold checkpoint fields do not match")
        if (state["version"] != 1 or state["num_classes"] != self.num_classes or
            state["initial_threshold"] != .6 or state["percentile"] != 60. or state["clip"] != [.5,.9]):
            raise ValueError("Pseudo-label threshold checkpoint settings do not match")
        count = np.asarray(state["count"])
        total = np.asarray(state["sum"],dtype=np.float64)
        if (count.shape != (self.num_classes,) or total.shape != count.shape or
            not np.issubdtype(count.dtype,np.integer) or np.any(count < 1) or
            not np.isfinite(total).all() or np.any(total < .6) or np.any(total > count)):
            raise ValueError("Invalid pseudo-label threshold accumulators")
        self.count = count.astype(np.int64,copy=True)
        self.confidence_sum = total.copy()

    def _endpoint(self, probabilities, update):
        confidence,prediction = probabilities.float().max(1)
        if update:
            flat_confidence = confidence.detach().cpu().numpy().reshape(-1)
            flat_prediction = prediction.detach().cpu().numpy().reshape(-1)
            for category in range(self.num_classes):
                assigned = flat_confidence[flat_prediction == category]
                if assigned.size:
                    self.confidence_sum[category] += float(np.percentile(assigned,60))*assigned.size
                    self.count[category] += assigned.size
        threshold = self.thresholds.copy()
        # Retain float64 threshold ties rather than rounding the cutoff down.
        cutoff = torch.as_tensor(threshold,device=probabilities.device)[prediction]
        return prediction,confidence >= cutoff,threshold

    @torch.no_grad()
    def generate(self, prob_pre, prob_post, label_pre, label_post, *,
                 valid=None, update_thresholds=True):
        if (prob_pre.ndim != 4 or prob_pre.shape != prob_post.shape or
            prob_pre.shape[1] != self.num_classes or prob_pre.device != prob_post.device):
            raise ValueError("Teacher probabilities must be matching BxCxHxW tensors")
        for probability in (prob_pre,prob_post):
            if (not probability.is_floating_point() or not bool(torch.isfinite(probability).all()) or
                bool(((probability < 0) | (probability > 1)).any()) or
                not torch.allclose(probability.float().sum(1),torch.ones_like(probability[:,0]).float(),atol=5e-3,rtol=0)):
                raise ValueError("Teacher inputs must be finite, normalized class probabilities")
        shape = prob_pre.shape[:1]+prob_pre.shape[2:]
        for label in (label_pre,label_post):
            if (label.shape != shape or label.dtype != torch.long or label.device != prob_pre.device or
                bool(((label < 0) | (label >= self.num_classes)).any())):
                raise ValueError("GT endpoint labels must be BxHxW int64 class indices")
        valid = torch.ones_like(label_pre,dtype=torch.bool) if valid is None else valid
        if valid.shape != shape or valid.dtype != torch.bool or valid.device != prob_pre.device:
            raise ValueError("valid must be a matching boolean tensor")
        prediction_pre,confident_pre,threshold_pre = self._endpoint(prob_pre,update_thresholds)
        prediction_post,confident_post,threshold_post = self._endpoint(prob_post,update_thresholds)
        candidates = valid & label_pre.eq(0) & label_post.eq(0)
        accepted = candidates & prediction_pre.eq(prediction_post) & prediction_pre.gt(0) & confident_pre & confident_post
        target_pre,target_post = label_pre.clone(),label_post.clone()
        target_pre[accepted],target_post[accepted] = prediction_pre[accepted],prediction_post[accepted]
        return {"target_pre":target_pre,"target_post":target_post,"accepted":accepted,
                "diagnostics":{"candidate_pixels":int(candidates.sum()),"accepted_pixels":int(accepted.sum()),
                               "thresholds_pre":threshold_pre.tolist(),"thresholds_post":threshold_post.tolist(),
                               "accepted_class_counts":torch.bincount(prediction_pre[accepted],minlength=self.num_classes).cpu().tolist()}}
