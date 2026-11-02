"""EC helpers and local accuracy_l1..l4 metric.

ASSUMPTION (official scorer not supplied): a top-1 prediction counts at level L if its L-prefix equals the
L-prefix of ANY recorded EC annotation of that protein."""
from __future__ import annotations


def prefix(ec, level):
    return ".".join(str(ec).split(".")[:level])


def accuracy_by_level(preds, truth):
    out = {}
    for L in (4, 3, 2, 1):
        hit = 0
        for k, true in truth.items():
            p = preds.get(k)
            if p and prefix(p[0], L) in {prefix(t, L) for t in true}:
                hit += 1
        out[f"accuracy_l{L}"] = hit / max(1, len(truth))
    return out
