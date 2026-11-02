"""Edge-case test input: duplicates, lowercase, whitespace, odd characters, empty, very short/long, plus mutated
(remote-homolog-like) proteins and optionally N random train proteins.  Writes a parquet with id, sequence.

    python scripts/make_test_input.py --out /tmp/edge.parquet [--n-mutated 40]"""
import argparse, random, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecpred.refdb import read_train_tsv

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--n-mutated", type=int, default=40)
ap.add_argument("--mut-rate", type=float, default=0.45); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed); AA = "ACDEFGHIKLMNPQRSTVWY"
tr = [s for _, s in read_train_tsv(ROOT / "weights")]
pick = rng.sample(tr, a.n_mutated + 5)
mut = lambda s: "".join(rng.choice(AA) if rng.random() < a.mut_rate else c for c in s)
rows = [mut(s) for s in pick[: a.n_mutated]]
edge = [pick[a.n_mutated], pick[a.n_mutated],                 # exact duplicate sequence (second id must be dropped)
        pick[a.n_mutated + 1].lower(), " " + pick[a.n_mutated + 2][:60] + "\n", "ACDXBZUO*-" * 6,
        "M", "", "MKV", "A" * 3000, pick[a.n_mutated + 3][:20]]
seqs = rows + edge
pd.DataFrame({"id": [f"seq_{i:04d}" for i in range(len(seqs))], "sequence": seqs}).to_parquet(a.out, index=False)
print(f"wrote {len(seqs)} rows ({len(seqs) - 1} unique) to {a.out}")
