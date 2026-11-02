"""'vote' scoring exactly as in the original (mode.json: scoring=vote): bitscore^power vote over the top-k hits,
down-weighting unreviewed sources; ties broken by training frequency then EC string."""
from __future__ import annotations


def rank_vote(hits, labels, freq, cfg):
    h = sorted(hits, key=lambda x: -x[5])[: int(cfg.vote_k)]
    if not h:
        return []
    top, votes = h[0][5], {}
    for s, _pid, _ql, _sl, _ev, b, *_ in h:
        ecs = labels.get(s)
        if not ecs:
            continue
        w = (b / top) ** float(cfg.vote_power) * cfg.source_weights.get(s[:3], 1.0)
        for e in ecs:
            votes[e] = votes.get(e, 0.0) + w / len(ecs)
    return sorted(votes, key=lambda e: (-votes[e], -freq.get(e, 0), e))[:10]


def rank_all(hits_by_query, labels, freq, cfg):
    out = {}
    for i in sorted(hits_by_query):
        r = rank_vote(hits_by_query[i], labels, freq, cfg)
        if r:
            out[i] = r
    return out
