"""Rebuild the DIAMOND databases from the packaged sources with the packaged binary.

    python scripts/rebuild_dbs.py [--out weights]

  train.dmnd        = train_seqs_*.tsv.gz  + data/swissprot_ext.fa.gz (sp_) + data/trembl_ext.fa.gz (un_)
  trembl_ev2.dmnd   = data/trembl_ev2_ext.fa.gz (u2_)
  train_only.dmnd   = train_seqs_*.tsv.gz  (strict mode, no external proteins)
The shipped train.dmnd has 281,143 sequences, i.e. exactly train (186,488) + sp (31,985) + un (62,670)."""
import argparse, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecpred.refdb import read_fasta_gz, read_train_tsv

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "weights")); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
dm = str(ROOT / "bin" / "diamond")
def build(name, records):
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        fa = Path(d) / "r.fa"
        with open(fa, "w") as f:
            n = 0
            for i, s in records:
                f.write(f">{i}\n{s}\n"); n += 1
        subprocess.run([dm, "makedb", "--in", str(fa), "-d", str(Path(a.out) / name), "--threads", str(a.threads),
                        "--quiet"], check=True)
    print(name, n, "sequences")
tr = list(read_train_tsv(ROOT / "weights"))
build("train_only", tr)
build("train", tr + list(read_fasta_gz(ROOT / "data/swissprot_ext.fa.gz")) + list(read_fasta_gz(ROOT / "data/trembl_ext.fa.gz")))
build("trembl_ev2", read_fasta_gz(ROOT / "data/trembl_ev2_ext.fa.gz"))
