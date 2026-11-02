# Enzyme EC prediction 

Task `leak-curber-enzyme-classification` · benchmark `leak-curber-enzyme-classification-20261023` · task `enzyme-classification`


## 1. Overview
Homology retrieval: every unique sequence is searched with DIAMOND against reference proteins that carry EC labels; the
EC numbers of the best hits are combined by a bitscore-weighted vote. This is the "v16 / vote" pipeline of the original
archive (`weights/mode.json`: `scoring=vote`), re-organised into modules. Pure numpy/pandas/scipy at inference.

## 2. Architecture / pipeline (`ecpred/pipeline.py`)
1. `prepare_inputs`: unique rows by sequence, first id wins (as the evaluator does). Sequences are never altered in the
   output; a sanitised copy (whitespace removed, non-letters → `X`) is used only as the DIAMOND query.
2. Smoke test (`--fast`, 20 queries) then main search vs `weights/train.dmnd` (`very-sensitive` for ≤14k queries,
   `sensitive` ≤40k, else `fast`), `-k 50`, chunked, 2 threads, config fallback `-c4 -b1` 1 thread.
3. Weak queries (no hit or best hit ≤30 % identity; ≤3000 of them; only if main search ended <215 s) are re-searched with `--ultra-sensitive`.
4. Extra fast pass vs `weights/trembl_ev2.dmnd` (only if elapsed <330 s); its hits are added to the pool.
5. Vote: `w = (bitscore/best)^12 × source weight` (`sp_` 1.0, `un_` 0.2, `u2_` 0.1) over the top 50 hits; ties → train EC
   frequency → EC string. Up to 10 distinct ECs returned, filtered to `ec_vocabulary.json`.
6. Safety nets: DIAMOND killed at 420 s (finished chunks kept); queries still unresolved get a 3-mer TF-IDF
   nearest-neighbour prediction (if <450 s elapsed); anything left gets the most frequent training EC.
`--scoring learned` switches to the shipped gradient-boosted re-ranker (`ranker.npz` + `te_tables.json.gz`); not the default.

Changes vs the original code (all other logic preserved): config errors are logged instead of silently swallowed; CNN
branch removed (its files `cnn.npz`/`cnn_numpy`/`cnn_feats` are not in the archive and `vote` mode never used them);
5 s CPU probe removed; `KeyError` guard for hit ids missing in labels; `--ignore-warnings` added to DIAMOND (a chunk
of only DNA-like sequences aborted the whole ultra-sensitive pass); partial results kept at the deadline; k-mer rescue
for unresolved queries; numpy ids converted for JSON.

## 3. Data / reference databases
`weights/train.dmnd` = 186,488 train + 31,985 Swiss-Prot (`sp_`) + 62,670 TrEMBL (`un_`) = 281,143 sequences.
`weights/trembl_ev2.dmnd` = 206,689 TrEMBL-v2 (`u2_`) sequences (**rebuilt** here; it was missing from the archive).
`weights/train_only.dmnd` = train only (for `--strict`). Rebuilding `train.dmnd` from the packaged sources with
`scripts/rebuild_dbs.py` reproduces the shipped file's sequence and letter counts exactly (281,143 / 135,059,194).
**External-data note:** UniProt-derived references can contain evaluation proteins verbatim. Check the task rules; use
`--strict` for a no-external-proteins run. Exact-match exclusion at build time is not applied to unseen test sequences
(they were not available).

## 4. Asset inventory
See `MANIFEST.md` (every file, size, role) and `CHECKSUMS.sha256`. DIAMOND v2.1.10 official binary, SHA-256 in `bin/NOTICE.txt`.
Missing and unrecoverable: CNN re-ranker files (unused), `train.parquet`/validation data (not supplied).

## 5. Install (local)
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
python -m pytest -q tests
```
## 6. Docker (not executed by the author — run these)
```bash
docker buildx build --platform linux/amd64 -t enzyme-ec:submission --load .
docker run --rm --platform linux/amd64 --network none --cpus 2 --memory 16g -v "$PWD/io":/io \
  enzyme-ec:submission python /app/predict.py --input /io/in.parquet --output /io/out.jsonl
python scripts/validate_output.py --input io/in.parquet --output io/out.jsonl
```
## 7. Competition command
`python /app/predict.py --input INPUT --output OUTPUT` (works directly; no entrypoint or shell init needed, no downloads).
Test hooks: `ECPRED_DISABLE_DIAMOND=1` (force k-mer fallback), `ECPRED_DEADLINE_S=<s>` (shorten the DIAMOND deadline).

## 8. Measured results (sandbox: Linux x86-64, **1 vCPU**, 3 GB RAM, venv with pinned requirements, Python 3.12)
| run | unique inputs | wall | peak RSS | result |
|---|---|---|---|---|
| default, 3,000 mutated train proteins (35 % substitutions) + 10 edge cases | 3,009 | 394 s | 1.43 GB | valid; 3,003 from neighbours, 6 default |
| `--strict`, 40 mutated + edge cases | 49 | 228 s | 2.43 GB | valid |
| DIAMOND disabled (k-mer fallback) | 49 | 25 s | 2.42 GB | valid |
| `ECPRED_DEADLINE_S=60` (watchdog path) | 49 | 80 s | 2.43 GB | valid; k-mer rescued 48/48 |
DIAMOND per call at 1 thread (fixed index cost + per query): fast ≈20 s; sensitive ≈93 s + 0.01 s/q; very-sensitive ≈84 s + 0.015 s/q;
ultra-sensitive ≈329 s + 0.14 s/q (`reports/diamond_mode_timing_1thread.log`). In the 3,009-query run the ultra pass cost
231 s for 5 weak queries and re-scored none, and it consumed the budget that would have allowed the extra TrEMBL-v2 pass.
Expect lower times on 2 vCPUs, but that is not measured. Queries were mutated copies of reference proteins, so these runs
validate speed and format, **not accuracy**.
Validator (`scripts/validate_output.py`) passed on all four outputs. Unit tests: 8 passed.

## 9. Accuracy
Not measured on the competition validation set (not supplied; Hugging Face unreachable from the build sandbox).
Reproduced: nothing. Previously reported by the original author: ~84 % L4 (unverified). A random hold-out of training
proteins gave ≈98.5 % L4, which is dominated by near-duplicates and is not comparable. Local scorer `scripts/evaluate.py`
assumes a prediction is correct at level L if it matches the prefix of ANY recorded EC (official scorer not supplied).
To measure: `python predict.py --input val.parquet --output o.jsonl && python scripts/evaluate.py --pred o.jsonl --labels val_labels.parquet`.

## 10. Known limitations
Docker image not built or run; 2-vCPU runtime and a full-size test set (size unknown) not measured; the heavy
ultra-sensitive pass has high fixed cost; the python base image is 3.11 while tests ran on 3.12; the numpy ranker mode
is untested here beyond unit checks; UniProt-reference leakage risk (§3).
