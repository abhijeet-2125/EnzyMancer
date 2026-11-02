"""Competition entry point:  python /app/predict.py --input INPUT.parquet --output OUTPUT.jsonl

Offline CPU inference. See README.md."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecpred.pipeline import predict_file  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strict", action="store_true", help="train-only references (needs weights/train_only.dmnd)")
    ap.add_argument("--scoring", choices=["vote", "learned"], help="override weights/mode.json")
    a = ap.parse_args()
    predict_file(a.input, a.output, a.strict, a.scoring)
