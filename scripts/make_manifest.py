"""Write CHECKSUMS.sha256 and MANIFEST.md (inventory + sizes) for the project directory."""
import hashlib, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
files = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
               and ".pytest_cache" not in p.parts and p.name not in ("CHECKSUMS.sha256", "MANIFEST.md"))
lines, rows, total = [], [], 0
for p in files:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    rel = p.relative_to(ROOT).as_posix(); sz = p.stat().st_size; total += sz
    lines.append(f"{h.hexdigest()}  {rel}"); rows.append((rel, sz))
(ROOT / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
USE = {"weights/train.dmnd": "DIAMOND DB: train + Swiss-Prot(sp_) + TrEMBL(un_), 281,143 seqs | inference | rebuild: scripts/rebuild_dbs.py",
       "weights/trembl_ev2.dmnd": "DIAMOND DB: TrEMBL v2 (u2_), 206,689 seqs; extra fast pass | inference | rebuild: scripts/rebuild_dbs.py",
       "weights/train_only.dmnd": "DIAMOND DB: train only, 186,488 seqs | inference with --strict | rebuild: scripts/rebuild_dbs.py",
       "weights/labels.json.gz": "id -> EC lists (all reference ids) + train EC frequencies | inference",
       "weights/ec_vocabulary.json": "legal EC vocabulary (6,939) | inference output filter",
       "weights/mode.json": "scoring mode + vote parameters | inference",
       "weights/ranker.npz": "gradient-boosted re-ranker (mode 'learned' only) | optional inference",
       "weights/te_tables.json.gz": "EC reliability tables for the re-ranker | optional inference",
       "weights/train_seqs_0.tsv.gz": "training sequences | k-mer fallback + DB rebuild",
       "weights/train_seqs_1.tsv.gz": "training sequences | k-mer fallback + DB rebuild",
       "bin/diamond": "DIAMOND v2.1.10 linux x86-64 | inference",
       "data/swissprot_ext.fa.gz": "external reference sequences (sp_) | DB rebuild only",
       "data/trembl_ext.fa.gz": "external reference sequences (un_) | DB rebuild only",
       "data/trembl_ev2_ext.fa.gz": "external reference sequences (u2_) | DB rebuild only"}
out = ["# MANIFEST", "", f"Total size: {total / 1e6:.1f} MB ({len(rows)} files)", "",
       "| file | bytes | role |", "|---|---:|---|"]
out += [f"| {r} | {s:,} | {USE.get(r, '')} |" for r, s in rows]
(ROOT / "MANIFEST.md").write_text("\n".join(out) + "\n")
print(f"{len(rows)} files, {total / 1e6:.1f} MB")
