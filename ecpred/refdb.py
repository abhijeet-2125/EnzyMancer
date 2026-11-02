"""Reference database: labels, FASTA writing, diamond makedb."""
from __future__ import annotations
import gzip
import json
import subprocess
from pathlib import Path


def read_fasta_gz(path):
    ident, buf = None, []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if ident is not None:
                    yield ident, "".join(buf)
                ident, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if ident is not None:
        yield ident, "".join(buf)


def read_train_tsv(weights):
    for part in sorted(Path(weights).glob("train_seqs_*.tsv.gz")):
        with gzip.open(part, "rt") as f:
            for line in f:
                i, s = line.rstrip("\n").split("\t")
                yield i, s


def load_labels(weights):
    """-> (labels: id -> [EC], freq: EC -> train count). freq comes from train only (tie-breaks)."""
    with gzip.open(Path(weights) / "labels.json.gz", "rt") as f:
        m = json.load(f)
    return m["labels"], m["freq"]


def write_reference_fasta(out_fa, train_items, external_items=(), exclude_seqs=frozenset()):
    """Write train + external records. Any record whose sequence is in `exclude_seqs`
    (test / held-out sequences) is dropped -> removes exact-match lookup leakage."""
    n_tr = n_ext = n_drop = 0
    with open(out_fa, "w") as f:
        for i, s in train_items:
            if s in exclude_seqs:
                continue
            f.write(f">{i}\n{s}\n"); n_tr += 1
        for i, s in external_items:
            if s in exclude_seqs:
                n_drop += 1
                continue
            f.write(f">{i}\n{s}\n"); n_ext += 1
    return {"train": n_tr, "external": n_ext, "external_dropped_exact": n_drop}


def makedb(diamond, fasta, db, threads=2):
    subprocess.run([diamond, "makedb", "--in", str(fasta), "-d", str(db), "--threads", str(threads), "--quiet"],
                   check=True)
