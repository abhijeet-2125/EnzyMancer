"""End-to-end inference (original v16 flow):

 1. DIAMOND blastp of every unique input vs train+Swiss-Prot+TrEMBL(reviewed-evidence) references.
 2. Weak queries (no hit / best hit <= weak_pid % identity) re-searched with --ultra-sensitive when time allows.
 3. Optional extra pass vs the larger TrEMBL v2 reference (fast mode); hits are only ADDED.
 4. Bitscore^power vote (mode.json scoring=vote) or the gradient-boosted re-ranker (scoring=learned).
 5. Safety net: DIAMOND smoke test + retries; if DIAMOND cannot run, a numpy/scipy 3-mer nearest-neighbour fallback;
    anything still unpredicted gets the most frequent training EC.
"""
from __future__ import annotations
import gzip
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import search
from .config import Config

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
START = time.time()


def log(msg):
    print(f"[{time.time() - START:7.1f}s] {msg}", file=sys.stderr, flush=True)


def diagnostics():
    try:
        log(f"python {sys.version.split()[0]} | {platform.platform()} | cpu_count={os.cpu_count()} "
            f"affinity={len(os.sched_getaffinity(0))} | /tmp free={shutil.disk_usage('/tmp').free / 1e9:.1f} GB")
    except Exception as exc:
        log(f"diagnostics: {exc!r}")


# ----------------------------------------------------------------------------------------------- inputs
def prepare_inputs(frame, task_json=ROOT / "task.json"):
    """Unique input rows exactly as the evaluator builds them: first id wins for identical sequences."""
    cols = ["id"] + json.loads(Path(task_json).read_text())["input_columns"]
    if not set(cols) <= set(frame.columns):
        raise ValueError("Missing input columns: " + ", ".join(cols))
    if frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError("Input IDs must be present and unique.")
    return frame[cols[:]].drop_duplicates(cols[1:], keep="first").reset_index(drop=True)


_BAD = re.compile(r"[^A-Za-z]")


def search_copy(seq):
    """Copy of a sequence made safe for a FASTA file (whitespace dropped, other non-letters -> X).
    Only the DIAMOND query uses it; output ids / predictions are never derived from modified text."""
    if not isinstance(seq, str):
        return ""
    return _BAD.sub("X", "".join(seq.split()))


# ----------------------------------------------------------------------------------------------- binaries
def find_diamond():
    if os.environ.get("ECPRED_DISABLE_DIAMOND"):                 # test hook: exercise the k-mer fallback
        return None
    exe = ROOT / "bin" / "diamond"
    if not exe.exists():
        return None
    if os.access(exe, os.X_OK):
        return str(exe)
    tmp = Path(tempfile.gettempdir()) / "diamond_bin" / "diamond"      # read-only / noexec-mode checkouts
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(exe, tmp)
    tmp.chmod(0o755)
    return str(tmp)


# ----------------------------------------------------------------------------------------------- homology
def homology_predictions(items, labels, freq, cfg, diamond):
    """items: [(query index, sequence)].  Returns ({query index: ranked ECs}, status) or (None, reason)."""
    from .scoring import rank_all
    n = len(items)
    main_mode = "very-sensitive" if n <= 14000 else ("sensitive" if n <= 40000 else "fast")
    db_main = WEIGHTS / ("train.dmnd" if cfg.use_external else "train_only.dmnd")
    if not db_main.exists():
        raise FileNotFoundError(db_main)
    seq_of = dict(items)
    workdir = tempfile.mkdtemp(dir="/tmp")
    hits, partial = None, False
    try:
        # --- smoke test + main search, most efficient config first
        sample = items[:20]
        for extra, threads in search.CONFIGS:
            try:
                search.run_diamond(diamond, db_main, search.write_chunks(sample, workdir, "s"), "fast", extra,
                                   threads, workdir, START, cfg.deadline_s, cfg.top_k)
                log(f"smoke test ok: {extra} t={threads}")
            except Exception as exc:
                log(f"smoke test failed: {exc}")
                continue
            try:
                hits = search.run_diamond(diamond, db_main, search.write_chunks(items, workdir, "a"), main_mode,
                                          extra, threads, workdir, START, cfg.deadline_s, cfg.top_k)
                break
            except search.DeadlineExceeded as exc:
                hits, partial = exc.hits, True
                log(f"main search hit the deadline: keeping {len(hits)} finished queries")
                break
            except Exception as exc:
                log(f"main search failed: {exc}")
                if time.time() - START > 240:
                    break
        if hits is None:
            return None, "diamond unavailable"
        t1 = time.time() - START
        weak = [i for i, _ in items if i not in hits or hits[i][0][1] <= cfg.weak_pid]
        log(f"main search ({main_mode}) done: {len(hits)}/{n} with hits, {len(weak)} weak")
        if not partial and weak and len(weak) <= cfg.weak_max and t1 < cfg.ultra_guard:
            try:
                ex = search.run_diamond(diamond, db_main, search.write_chunks(((i, seq_of[i]) for i in weak),
                                        workdir, "b"), "ultra-sensitive", extra, threads, workdir, START,
                                        cfg.deadline_s, cfg.top_k)
                hits.update(ex)
                log(f"ultra-sensitive pass done: {len(ex)} re-scored")
            except Exception as exc:
                log(f"ultra-sensitive pass skipped: {exc}")
        t2 = time.time() - START
        xdb = WEIGHTS / cfg.extra_db if cfg.extra_db else None
        if cfg.use_external and xdb and xdb.exists() and not partial and t2 < cfg.extra_guard:
            try:
                ex = search.run_diamond(diamond, xdb, search.write_chunks(items, workdir, "x"), cfg.extra_mode,
                                        extra, threads, workdir, START, cfg.deadline_s, cfg.top_k)
                for i, h in ex.items():
                    hits.setdefault(i, []).extend(h)
                log(f"extra reference pass ({cfg.extra_db}) done: {len(ex)} with hits, {time.time() - START - t2:.0f}s")
            except Exception as exc:
                log(f"extra reference pass skipped: {exc}")
        elif cfg.extra_db:
            log(f"extra reference pass skipped (elapsed {t2:.0f}s)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    log(f"scoring: {cfg.scoring}")
    if cfg.scoring == "vote":
        return rank_all(hits, labels, freq, cfg), "diamond"
    return _learned(hits, labels, freq, cfg), "diamond"


def _learned(hits, labels, freq, cfg):
    from .features import query_candidates
    from .ranker import EcReliability, TreeRanker
    ranker = TreeRanker(WEIGHTS / "ranker.npz")
    rel = EcReliability(WEIGHTS / "te_tables.json.gz")
    order, cand_of, blocks = sorted(hits), {}, []
    for i in order:
        cands, X = query_candidates(sorted(hits[i], key=lambda x: -x[5]), labels, freq)
        if cands:
            cand_of[i] = cands
            blocks.append(np.hstack([X.astype(np.float64), rel.matrix(cands)]))
    scores = ranker.score(np.vstack(blocks)) if blocks else np.zeros(0)
    preds, pos = {}, 0
    for i in order:
        if i in cand_of:
            c = cand_of[i]
            s = scores[pos: pos + len(c)]
            pos += len(c)
            preds[i] = [c[j] for j in np.argsort(-s, kind="stable")]
    return preds


def kmer_predictions(items, labels, freq):
    """Fallback without DIAMOND: 3-mer TF-IDF nearest neighbours."""
    from .kmer import KmerIndex
    ids, tseqs = [], []
    for part in sorted(WEIGHTS.glob("train_seqs_*.tsv.gz")):
        with gzip.open(part, "rt") as f:
            for line in f:
                i, s = line.rstrip("\n").split("\t")
                ids.append(i)
                tseqs.append(s)
    log(f"k-mer fallback: {len(tseqs)} training sequences")
    idx = KmerIndex(tseqs)
    del idx.XT, tseqs
    qi = [i for i, _ in items]
    I, S = idx.query_fast([s.upper() for _, s in items])
    preds = {}
    for q, (row, sc) in zip(qi, zip(I, S)):
        votes = {}
        for j, s in zip(row, sc):
            ecs = labels[ids[j]]
            for e in ecs:
                votes[e] = votes.get(e, 0.0) + max(float(s), 0.0) ** 4 / len(ecs)
        if votes:
            preds[q] = sorted(votes, key=lambda e: (-votes[e], -freq.get(e, 0), e))[:10]
    return preds


# ----------------------------------------------------------------------------------------------- entry
def predict_file(input_path, output_path, strict=False, scoring=None):
    global START
    START = time.time()
    diagnostics()
    cfg = Config.load(WEIGHTS / "mode.json", log)
    cfg.use_external = cfg.use_external and not strict
    if os.environ.get("ECPRED_DEADLINE_S"):                        # test hook: exercise the deadline path
        cfg.deadline_s = float(os.environ["ECPRED_DEADLINE_S"])
    if scoring:
        cfg.scoring = scoring
    frame = prepare_inputs(pd.read_parquet(input_path))
    with gzip.open(WEIGHTS / "labels.json.gz", "rt") as f:
        model = json.load(f)
    labels, freq = model["labels"], model["freq"]
    vocab = set(json.loads((WEIGHTS / "ec_vocabulary.json").read_text()))
    default = sorted(freq, key=lambda e: (-freq[e], e))[0]
    seqs = list(frame.sequence)
    items = [(i, search_copy(s)) for i, s in enumerate(seqs)]
    items = [(i, s) for i, s in items if s]            # empty / non-string sequences cannot be searched
    log(f"{len(seqs)} unique inputs ({len(items)} searchable)")

    preds, method = None, "diamond"
    diamond = find_diamond()
    if items and diamond:
        try:
            preds, why = homology_predictions(items, labels, freq, cfg, diamond)
            if preds is None:
                log(f"homology pipeline unavailable: {why}")
        except Exception as exc:
            log(f"homology pipeline error: {exc!r}")
    elif items:
        log("bin/diamond unavailable")
    if preds is None and items:
        method = "kmer-fallback"
        try:
            preds = kmer_predictions(items, labels, freq)
        except Exception as exc:
            log(f"k-mer fallback error: {exc!r}")
            method = "frequency-only"
    preds = preds or {}
    missing = [(i, sq) for i, sq in items if not preds.get(i)]
    if missing and method == "diamond" and time.time() - START < 450:
        # queries with no usable hit (or lost to the deadline): 3-mer neighbours beat a constant guess
        try:
            extra = kmer_predictions(missing, labels, freq)
            preds.update(extra)
            log(f"k-mer rescue: {len(extra)}/{len(missing)} unresolved queries")
        except Exception as exc:
            log(f"k-mer rescue skipped: {exc!r}")

    n_fallback = 0
    with Path(output_path).open("w") as out:
        for i, key in enumerate(frame.id):
            ranked = [e for e in dict.fromkeys(preds.get(i) or []) if e in vocab][:10]
            if not ranked:
                ranked, n_fallback = [default], n_fallback + 1
            key = key.item() if hasattr(key, "item") else key           # numpy scalar -> python
            out.write(json.dumps({"id": key, "prediction": ranked}) + "\n")
    log(f"done: method={method}, {len(frame) - n_fallback}/{len(frame)} predicted from neighbours, "
        f"{n_fallback} default-EC fallbacks")
    return {"method": method, "unique": len(frame), "fallback_default": n_fallback}
