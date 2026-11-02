"""Learned re-ranker pieces from the original (mode 'learned'): numpy tree-ensemble scorer + EC reliability tables.
sklearn is only needed by export_hgb()/fit (training), never at inference."""
from __future__ import annotations
import gzip
import json
import numpy as np


class TreeRanker:
    def __init__(self, path):
        z = np.load(path)
        self.feat, self.thr = z["feat"], z["thr"]
        self.left, self.right = z["left"], z["right"]
        self.leaf, self.value = z["leaf"], z["value"]
        self.roots, self.depth = z["roots"], int(z["depth"])
        self.base = float(z["base"])

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        out = np.full(n, self.base)
        rows = np.arange(n)
        for root in self.roots:
            idx = np.full(n, root)
            for _ in range(self.depth):
                go_left = X[rows, self.feat[idx]] <= self.thr[idx]
                nxt = np.where(go_left, self.left[idx], self.right[idx])
                idx = np.where(self.leaf[idx], idx, nxt)
            out += self.value[idx]
        return out


class EcReliability:
    def __init__(self, path):
        with gzip.open(path, "rt") as f:
            t = json.load(f)
        self.t, self.A = t, t["A"]

    def _enc(self, table, key, prior):
        s, c = self.t[table].get(key, (0.0, 0))
        return (s + self.A * prior) / (c + self.A), np.log1p(c)

    def matrix(self, cands):
        rows = []
        for e in cands:
            a, an = self._enc("ec", e, self.t["prior"])
            b, bn = self._enc("ec_top", e, self.t["prior_top"])
            c, cn = self._enc("l3", e.rsplit(".", 1)[0], self.t["prior"])
            rows.append([a, an, b, bn, c, cn])
        return np.array(rows, dtype=np.float64)
