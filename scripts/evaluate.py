"""Local scoring of a predictions JSONL against labels.

    python scripts/evaluate.py --pred out.jsonl --labels val_labels.parquet     (columns: id, ec_number; repeated ids allowed)

Multi-annotation rule (ASSUMED, official scorer not supplied): top-1 is correct at level L if its L-prefix equals the
L-prefix of any recorded EC of that protein."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecpred.ec import accuracy_by_level

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True); ap.add_argument("--labels", required=True)
a = ap.parse_args()
preds = {}
for l in open(a.pred):
    r = json.loads(l); preds[str(r["id"])] = r["prediction"]
lab = pd.read_parquet(a.labels)
truth = defaultdict(list)
for i, e in zip(lab.id.astype(str), lab.ec_number):
    truth[i] += list(e) if isinstance(e, (list, tuple)) else [e]
truth = {k: v for k, v in truth.items() if k in preds}
print(json.dumps({"n_scored": len(truth), **accuracy_by_level(preds, truth)}, indent=2))
