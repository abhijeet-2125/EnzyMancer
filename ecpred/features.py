"""Candidate-EC features from DIAMOND hits (shared by training and inference)."""
import math
import numpy as np

MAX_CAND = 10
FEATURES = [
    'v1', 'v4', 'v8', 'vrank', 'best_bits_rel', 'best_bits', 'best_pid', 'best_rank',
    'cnt10', 'cnt50', 'frac10', 'qcov', 'scov', 'log_ev', 'best_nec', 'l3', 'l2', 'l1',
    'log_freq', 'top_bits', 'top_pid', 'log_qlen', 'n_hits', 'n_cand', 'top_nec',
    'margin4', 'pid_gap',
]


def query_candidates(hits, labels, freq, source_weights=None):
    """hits: list of (sseqid, pid, qlen, slen, evalue, bits, qstart, qend, sstart, send),
    sorted by bitscore descending. Returns (candidate ECs, feature matrix)."""
    hits = [h for h in hits if labels.get(h[0])]     # v16 raised KeyError on ids missing from labels
    if not hits:
        return [], np.zeros((0, len(FEATURES)), dtype=np.float32)
    top_bits = hits[0][5]
    top_pid = hits[0][1]
    qlen = hits[0][2]
    stats = {}
    tot1 = tot4 = tot8 = 0.0
    lvl = [dict(), dict(), dict()]  # L3, L2, L1 prefix votes (b^4)
    for r, (s, pid, ql, sl, ev, b, qs, qe, ss, se) in enumerate(hits):
        ecs = labels[s]
        n = len(ecs)
        rb = b / top_bits
        w1, w4, w8 = rb / n, rb ** 4 / n, rb ** 8 / n
        tot1 += rb
        tot4 += rb ** 4
        tot8 += rb ** 8
        prefixes = [set(), set(), set()]
        for e in ecs:
            st = stats.get(e)
            if st is None:
                st = stats[e] = [0.0, 0.0, 0.0, b, pid, r, 0, 0,
                                 (qe - qs + 1) / ql, (se - ss + 1) / sl, ev, n]
            st[0] += w1
            st[1] += w4
            st[2] += w8
            if r < 10:
                st[6] += 1
            st[7] += 1
            p = e.split('.')
            for L in range(3):
                prefixes[L].add('.'.join(p[:3 - L]))
        for L in range(3):
            for pre in prefixes[L]:
                lvl[L][pre] = lvl[L].get(pre, 0.0) + rb ** 4
    cands = sorted(stats, key=lambda e: (-stats[e][1], -freq.get(e, 0), e))[:MAX_CAND]
    v4s = [stats[e][1] for e in cands]
    margin = (v4s[0] - v4s[1]) / tot4 if len(v4s) > 1 else v4s[0] / tot4
    top_nec = len(labels[hits[0][0]])
    n10 = min(10, len(hits))
    X = np.zeros((len(cands), len(FEATURES)), dtype=np.float32)
    for i, e in enumerate(cands):
        st = stats[e]
        p = e.split('.')
        X[i] = [st[0] / tot1, st[1] / tot4, st[2] / tot8, i, st[3] / top_bits, st[3], st[4], st[5],
                st[6], st[7], st[6] / n10, st[8], st[9], math.log10(st[10] + 1e-180), st[11],
                lvl[0]['.'.join(p[:3])] / tot4, lvl[1]['.'.join(p[:2])] / tot4,
                lvl[2][p[0]] / tot4, math.log1p(freq.get(e, 0)), top_bits, top_pid,
                math.log(qlen), len(hits), len(stats), top_nec, margin, top_pid - st[4]]
    return cands, X


def parse_line(line):
    q, s, pid, length, ql, sl, ev, b, qs, qe, ss, se = line.rstrip('\n').split('\t')
    return q, (s, float(pid), int(ql), int(sl), float(ev), float(b), int(qs), int(qe), int(ss), int(se))
