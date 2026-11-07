"""Streaming pixel metrics and token-based caption evaluation."""
from __future__ import annotations

from collections import Counter
import math
import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _mask(value):
    value = _numpy(value)
    return value[:,0] if value.ndim == 4 and value.shape[1] == 1 else value


def _discrete(value, classes, name):
    value = _mask(value)
    if not np.issubdtype(value.dtype, np.integer) and value.dtype != np.bool_:
        if not np.equal(value, np.round(value)).all():
            raise ValueError(f"{name} must contain discrete labels")
    if not np.isin(value, range(classes)).all():
        raise ValueError(f"{name} labels must lie in 0..{classes-1}")
    return value.astype(np.int64, copy=False)


class BinaryMetrics:
    def __init__(self):
        self.hist = np.zeros((2,2), dtype=np.int64)

    def update(self, prediction, target, valid_mask=None):
        p, y = _mask(prediction), _mask(target)
        valid = np.ones(y.shape, bool) if valid_mask is None else _mask(valid_mask).astype(bool)
        if p.shape != y.shape or valid.shape != y.shape:
            raise ValueError("Predictions, targets, and validity mask must share their shape")
        p, y = _discrete(p[valid],2,"prediction"), _discrete(y[valid],2,"target")
        self.hist += np.bincount(2*y+p, minlength=4).reshape(2,2)

    def compute(self):
        tn, fp, fn, tp = map(int,self.hist.ravel())
        ratio = lambda a,b: float(a/b) if b else 0.
        return {"F1":ratio(2*tp,2*tp+fp+fn), "IoU":ratio(tp,tp+fp+fn),
                "precision":ratio(tp,tp+fp), "recall":ratio(tp,tp+fn),
                "accuracy":ratio(tp+tn,self.hist.sum()), "confusion":self.hist.tolist()}


class SemanticMetrics:
    def __init__(self, num_classes=7):
        if num_classes < 2:
            raise ValueError("SCD needs unchanged class 0 plus semantic classes")
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes,num_classes), dtype=np.int64)

    def update(self, pred_pre, pred_post, label_pre, label_post, valid_mask=None):
        for prediction,target in ((pred_pre,label_pre),(pred_post,label_post)):
            p, y = _mask(prediction), _mask(target)
            valid = np.ones(y.shape,bool) if valid_mask is None else _mask(valid_mask).astype(bool)
            if p.shape != y.shape or valid.shape != y.shape:
                raise ValueError("SCD predictions, labels, and mask shapes differ")
            p, y = _discrete(p[valid],self.num_classes,"prediction"), _discrete(y[valid],self.num_classes,"target")
            self.hist += np.bincount(self.num_classes*p+y, minlength=self.num_classes**2).reshape(self.num_classes,self.num_classes)

    def compute(self):
        hist = self.hist.astype(np.float64)
        binary = np.array([[hist[0,0],hist[0,1:].sum()], [hist[1:,0].sum(),hist[1:,1:].sum()]])
        denom = binary.sum(0)+binary.sum(1)-np.diag(binary)
        iou = np.divide(np.diag(binary),denom,out=np.full(2,np.nan),where=denom>0)
        separated = hist.copy(); separated[0,0] = 0
        n = separated.sum()
        if n:
            po = np.trace(separated)/n
            pe = np.dot(separated.sum(0),separated.sum(1))/(n*n)
            kappa = (po-pe)/(1-pe) if pe != 1 else 0.
        else:
            kappa = 0.
        sc_tp = np.trace(hist[1:,1:])
        return {"mIoU":float(iou.mean()), "SeK":float(kappa*math.exp(iou[1]-1)),
                "Fscd":float(2*sc_tp/max(hist[1:,:].sum()+hist[:,1:].sum(),1)),
                "OA":float(np.trace(hist)/hist.sum()) if hist.sum() else float("nan"),
                "confusion":self.hist.tolist()}


class DamageMetrics:
    """Localization F1 and harmonic damage F1 at known GT building pixels."""
    def __init__(self):
        self.localization = BinaryMetrics()
        self.damage = np.zeros((5,5), dtype=np.int64)

    def update(self, pred_localization, pred_damage, localization, damage, valid_mask=None):
        pl, pd, loc, dmg = [_mask(x) for x in (pred_localization,pred_damage,localization,damage)]
        if any(x.shape != loc.shape for x in (pl,pd,dmg)):
            raise ValueError("BDA predictions and targets must share their shape")
        valid = loc != 255
        if valid_mask is not None:
            valid &= _mask(valid_mask).astype(bool)
        self.localization.update(pl,loc,valid)
        selected = valid & (loc == 1) & (dmg != 255)
        y = _discrete(dmg[selected],5,"damage target")
        p = _discrete(pd[selected],5,"damage prediction")
        self.damage += np.bincount(5*y+p,minlength=25).reshape(5,5)

    def compute(self):
        loc = self.localization.compute()["F1"]
        tp = np.diag(self.damage)[1:].astype(float)
        fn, fp = self.damage.sum(1)[1:]-tp, self.damage.sum(0)[1:]-tp
        precision = tp/(tp+fp+1e-7)
        recall = tp/(tp+fn+1e-7)
        f1 = 2*precision*recall/(precision+recall+1e-7)
        harmonic = float(4/np.sum(1/f1)) if np.all(f1 > 0) else 0.
        return {"F1_loc":loc,"F1_cls":harmonic,"F1_overall":.3*loc+.7*harmonic,
                "damage_F1":dict(zip(("no_damage","minor_damage","major_damage","destroyed"),map(float,f1))),
                "loc_confusion":self.localization.hist.tolist(),"damage_confusion":self.damage.tolist()}


def _tokens(value):
    value = value.split() if isinstance(value,str) else list(value)
    if any(not isinstance(x,str) or not x or any(c.isspace() for c in x) for x in value):
        raise ValueError("Caption metrics take token lists or whitespace-separated sentences")
    if any(x in ("<start>","<end>","<pad>") for x in value):
        raise ValueError("Remove decoder boundary and padding tokens before scoring")
    return value


def _lcs(a,b):
    row = [0]*(len(b)+1)
    for x in a:
        last = 0
        for j,y in enumerate(b,1):
            old = row[j]
            row[j] = last+1 if x == y else max(row[j],row[j-1])
            last = old
    return row[-1]


def caption_scores(references, hypotheses, *, backend="builtin", include_meteor=False):
    """Token scores in native units.

    ``builtin`` computes unsmoothed corpus BLEU-1..4 and multi-reference
    ROUGE-L (beta=1.2). ``coco`` uses pycocoevalcap for its official BLEU,
    ROUGE-L, CIDEr, and optionally METEOR implementations.
    """
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Caption scoring needs matching nonempty references and hypotheses")
    refs = [[_tokens(x) for x in group] for group in references]
    hyps = [_tokens(x) for x in hypotheses]
    if any(not group for group in refs):
        raise ValueError("Each image pair needs at least one reference")
    if backend == "coco":
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider
        except ImportError as error:
            raise ImportError("Install the project's [caption] extra for COCO caption metrics") from error
        r = {i:[" ".join(x) for x in group] for i,group in enumerate(refs)}
        h = {i:[" ".join(x)] for i,x in enumerate(hyps)}
        bleu,_ = Bleu(4).compute_score(r,h)
        rouge,_ = Rouge().compute_score(r,h)
        cider,_ = Cider().compute_score(r,h)
        result = {f"Bleu_{i+1}":float(value) for i,value in enumerate(bleu)}
        result.update(ROUGE_L=float(rouge),CIDEr=float(cider))
        if include_meteor:
            from pycocoevalcap.meteor.meteor import Meteor
            meteor = Meteor()
            try:
                result["METEOR"] = float(meteor.compute_score(r,h)[0])
            finally:
                if hasattr(meteor,"close"):
                    meteor.close()
        return result
    if backend != "builtin":
        raise ValueError("Caption metric backend must be builtin or coco")
    if include_meteor:
        raise ValueError("METEOR requires backend='coco' and Java")
    clipped = np.zeros(4,dtype=float)
    total = np.zeros(4,dtype=float)
    hypothesis_length = reference_length = 0
    rouge_scores = []
    for group,hyp in zip(refs,hyps):
        hypothesis_length += len(hyp)
        reference_length += min((len(x) for x in group),key=lambda n:(abs(n-len(hyp)),n))
        for n in range(1,5):
            count = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
            maxima = Counter()
            for ref in group:
                maxima |= Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            clipped[n-1] += sum(min(value,maxima[key]) for key,value in count.items())
            total[n-1] += sum(count.values())
        precision = max((_lcs(hyp,ref)/len(hyp) if hyp else 0.) for ref in group)
        recall = max((_lcs(hyp,ref)/len(ref) if ref else 0.) for ref in group)
        beta2 = 1.2**2
        rouge_scores.append((1+beta2)*precision*recall/(recall+beta2*precision) if recall+precision else 0.)
    p = np.divide(clipped,total,out=np.zeros(4),where=total>0)
    bp = math.exp(min(0.,1-reference_length/hypothesis_length)) if hypothesis_length else 0.
    result = {f"Bleu_{n}":float(bp*math.exp(float(np.log(p[:n]).mean()))) if np.all(p[:n] > 0) else 0.
              for n in range(1,5)}
    result["ROUGE_L"] = float(np.mean(rouge_scores))
    return result


class CaptionMetrics:
    def __init__(self):
        self.references, self.hypotheses = [], []

    def update(self,references,hypotheses):
        if len(references) != len(hypotheses):
            raise ValueError("Caption references and hypotheses have different counts")
        self.references.extend(references)
        self.hypotheses.extend(hypotheses)

    def compute(self, *, backend="builtin", include_meteor=False):
        return caption_scores(self.references,self.hypotheses,backend=backend,include_meteor=include_meteor)


def create_metric(task: str, num_classes: int = 7):
    if task == "bcd":
        return BinaryMetrics()
    if task == "scd":
        return SemanticMetrics(num_classes)
    if task == "bda":
        return DamageMetrics()
    if task == "cc":
        return CaptionMetrics()
    raise ValueError(f"Unknown task {task!r}")


def update_metrics(task: str, meter, out: dict, batch: dict, *, wordmap=None, threshold=.5):
    valid = batch.get("valid_mask")
    if task == "bcd":
        meter.update(out["change_logits"].detach().sigmoid() > threshold,batch["label"],valid)
    elif task == "scd":
        changed = out["pred_cd"].detach()[:,0] > threshold
        a,b = [out[key].detach().argmax(1)*changed for key in ("pred_a","pred_b")]
        meter.update(a,b,batch["label_pre"],batch["label_post"],valid)
    elif task == "bda":
        loc = out.get("localization_logits",out.get("loc_logits"))
        meter.update(loc.detach().sigmoid() > threshold,out["damage_logits"].detach().argmax(1),
                     batch["localization"],batch["damage"],valid)
    elif task == "cc":
        hypotheses = out.get("hypotheses")
        if hypotheses is None:
            if wordmap is None or "generated_tokens" not in out:
                raise ValueError("Caption metric updates need hypotheses, or generated_tokens and wordmap")
            from .data import decode_caption
            hypotheses = [decode_caption(row,wordmap) for row in out["generated_tokens"]]
        meter.update(batch["references"],hypotheses)
    else:
        raise ValueError(f"Unknown task {task!r}")
