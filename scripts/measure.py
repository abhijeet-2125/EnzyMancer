"""Run predict.py as the evaluator does and report wall time, CPU time, peak RSS (all child processes) and output size.

    python scripts/measure.py --input in.parquet --output out.jsonl [-- extra predict.py args]"""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
t0 = time.time()
rc = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--input", a.input, "--output", a.output,
                     *[x for x in a.rest if x != "--"]]).returncode
wall = time.time() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(json.dumps({"returncode": rc, "wall_s": round(wall, 1), "cpu_user_s": round(ru.ru_utime, 1),
                  "cpu_sys_s": round(ru.ru_stime, 1), "cpu_util_pct": round(100 * (ru.ru_utime + ru.ru_stime) / wall),
                  "peak_rss_mb_largest_child": round(ru.ru_maxrss / 1024),
                  "output_bytes": os.path.getsize(a.output) if os.path.exists(a.output) else None}))
sys.exit(rc)
