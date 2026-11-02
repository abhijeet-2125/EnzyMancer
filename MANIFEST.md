# MANIFEST

Total size: 546.2 MB (47 files)

| file | bytes | role |
|---|---:|---|
| .dockerignore | 78 |  |
| Dockerfile | 802 |  |
| README.md | 6,828 |  |
| bin/NOTICE.txt | 425 |  |
| bin/diamond | 104,746,776 | DIAMOND v2.1.10 linux x86-64 | inference |
| data/swissprot_ext.fa.gz | 7,518,168 | external reference sequences (sp_) | DB rebuild only |
| data/trembl_ev2_ext.fa.gz | 39,410,923 | external reference sequences (u2_) | DB rebuild only |
| data/trembl_ext.fa.gz | 11,869,872 | external reference sequences (un_) | DB rebuild only |
| ecpred/__init__.py | 135 |  |
| ecpred/config.py | 1,834 |  |
| ecpred/ec.py | 652 |  |
| ecpred/features.py | 3,031 |  |
| ecpred/kmer.py | 3,626 |  |
| ecpred/pipeline.py | 11,703 |  |
| ecpred/ranker.py | 1,785 |  |
| ecpred/refdb.py | 1,977 |  |
| ecpred/scoring.py | 962 |  |
| ecpred/search.py | 4,616 |  |
| predict.py | 771 |  |
| reports/diamond_mode_timing_1thread.log | 378 |  |
| reports/run_3009_default.json | 153 |  |
| reports/run_3009_default.log | 505 |  |
| reports/run_edge49_prefix.json | 151 |  |
| reports/run_edge49_prefix.log | 692 |  |
| reports/run_edge49_strict.json | 151 |  |
| reports/run_edge49_strict.log | 594 |  |
| requirements-dev.txt | 27 |  |
| requirements.txt | 57 |  |
| scripts/evaluate.py | 1,111 |  |
| scripts/make_manifest.py | 2,486 |  |
| scripts/make_test_input.py | 1,504 |  |
| scripts/measure.py | 1,204 |  |
| scripts/original_build_external_reference.py | 3,948 |  |
| scripts/rebuild_dbs.py | 1,580 |  |
| scripts/validate_output.py | 2,371 |  |
| task.json | 480 |  |
| tests/test_core.py | 3,183 |  |
| weights/ec_vocabulary.json | 100,196 | legal EC vocabulary (6,939) | inference output filter |
| weights/labels.json.gz | 2,809,589 | id -> EC lists (all reference ids) + train EC frequencies | inference |
| weights/mode.json | 180 | scoring mode + vote parameters | inference |
| weights/ranker.npz | 632,858 | gradient-boosted re-ranker (mode 'learned' only) | optional inference |
| weights/te_tables.json.gz | 33,398 | EC reliability tables for the re-ranker | optional inference |
| weights/train.dmnd | 143,298,378 | DIAMOND DB: train + Swiss-Prot(sp_) + TrEMBL(un_), 281,143 seqs | inference | rebuild: scripts/rebuild_dbs.py |
| weights/train_only.dmnd | 84,259,718 | DIAMOND DB: train only, 186,488 seqs | inference with --strict | rebuild: scripts/rebuild_dbs.py |
| weights/train_seqs_0.tsv.gz | 21,706,308 | training sequences | k-mer fallback + DB rebuild |
| weights/train_seqs_1.tsv.gz | 22,870,147 | training sequences | k-mer fallback + DB rebuild |
| weights/trembl_ev2.dmnd | 106,860,680 | DIAMOND DB: TrEMBL v2 (u2_), 206,689 seqs; extra fast pass | inference | rebuild: scripts/rebuild_dbs.py |
