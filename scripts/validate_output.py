"""Competition-format validator.

    python scripts/validate_output.py --input in.parquet --output out.jsonl [--weights weights]

Checks: one row per UNIQUE sequence (first id kept), exact ids, no extras / duplicates, schema exactly
{"id","prediction"}, 1..10 predictions, all in the EC vocabulary, no repeated EC in a row, valid JSON per line."""
import argparse, json, sys
from pathlib import Path
import pandas as pd


def validate(input_path, output_path, weights="weights"):
    errs = []
    vocab = set(json.loads((Path(weights) / "ec_vocabulary.json").read_text()))
    df = pd.read_parquet(input_path)
    expected = list(df.drop_duplicates("sequence", keep="first").id)
    seen, n = [], 0
    for ln, line in enumerate(Path(output_path).read_text().splitlines(), 1):
        n += 1
        try:
            row = json.loads(line)
        except Exception as exc:
            errs.append(f"line {ln}: invalid JSON ({exc})"); continue
        if set(row) != {"id", "prediction"}:
            errs.append(f"line {ln}: keys {sorted(row)} != ['id','prediction']")
        seen.append(row.get("id"))
        p = row.get("prediction")
        if not isinstance(p, list) or not 1 <= len(p) <= 10:
            errs.append(f"line {ln}: prediction must be a list of 1..10 ECs"); continue
        if len(set(p)) != len(p):
            errs.append(f"line {ln}: duplicate EC in prediction")
        bad = [e for e in p if e not in vocab]
        if bad:
            errs.append(f"line {ln}: EC outside vocabulary: {bad[:3]}")
    if len(seen) != len(set(map(str, seen))):
        errs.append("duplicate ids in output")
    miss, extra = set(map(str, expected)) - set(map(str, seen)), set(map(str, seen)) - set(map(str, expected))
    if miss: errs.append(f"{len(miss)} unique input ids missing, e.g. {sorted(miss)[:3]}")
    if extra: errs.append(f"{len(extra)} extra ids, e.g. {sorted(extra)[:3]}")
    return {"rows": n, "expected_unique": len(expected), "errors": errs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parents[1] / "weights"))
    a = ap.parse_args()
    r = validate(a.input, a.output, a.weights)
    print(json.dumps(r, indent=2)[:3000])
    sys.exit(1 if r["errors"] else 0)
