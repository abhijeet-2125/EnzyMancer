import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecpred import pipeline
from ecpred.config import Config
from ecpred.ec import accuracy_by_level, prefix
from ecpred.features import query_candidates, FEATURES
from ecpred.scoring import rank_vote

def hit(s, b, pid=50.0): return (s, pid, 100, 100, 1e-20, b, 1, 100, 1, 100)
LABELS = {"a": ["1.1.1.1"], "b": ["1.1.1.2"], "sp_x": ["2.7.1.1"], "un_y": ["3.1.1.1"]}
FREQ = {"1.1.1.1": 5, "1.1.1.2": 4, "2.7.1.1": 9, "3.1.1.1": 1}

def test_metric_multi_annotation():
    r = accuracy_by_level({"q": ["1.2.3.9"], "p": ["4.1.1.1"]}, {"q": ["1.2.3.4", "9.9.9.9"], "p": ["4.1.1.1"]})
    assert r["accuracy_l3"] == 1.0 and r["accuracy_l4"] == 0.5

def test_vote_order_and_ties():
    cfg = Config(vote_power=1.0)
    assert rank_vote([hit("a", 100), hit("b", 90)], LABELS, FREQ, cfg)[0] == "1.1.1.1"
    # equal votes -> higher training frequency first
    assert rank_vote([hit("a", 100), hit("b", 100)], LABELS, FREQ, cfg)[:2] == ["1.1.1.1", "1.1.1.2"]

def test_source_weight_downweights_unreviewed():
    cfg = Config(vote_power=0.0)           # weight 1 each, un_ = 0.2
    assert rank_vote([hit("un_y", 100), hit("sp_x", 90)], LABELS, FREQ, cfg)[0] == "2.7.1.1"

def test_unknown_subject_and_no_hits():
    assert query_candidates([hit("zzz", 90), hit("a", 80)], LABELS, FREQ)[0] == ["1.1.1.1"]
    assert query_candidates([], LABELS, FREQ)[0] == []
    assert rank_vote([hit("zzz", 10)], LABELS, FREQ, Config()) == []

def test_search_copy_never_empties_valid_sequences():
    assert pipeline.search_copy("mk v\n") == "mkv" and pipeline.search_copy("AC*-1") == "ACXXX"
    assert pipeline.search_copy(None) == "" and pipeline.search_copy("   ") == ""

def test_prepare_inputs_first_id_wins():
    df = pd.DataFrame({"id": ["a", "b", "c"], "sequence": ["MKV", "MKV", "MKVL"]})
    assert list(pipeline.prepare_inputs(df).id) == ["a", "c"]
    with pytest.raises(ValueError): pipeline.prepare_inputs(pd.DataFrame({"id": ["a", "a"], "sequence": ["X", "Y"]}))
    with pytest.raises(ValueError): pipeline.prepare_inputs(pd.DataFrame({"id": ["a"]}))

def test_assets_consistent():
    W = ROOT / "weights"
    for f in ["labels.json.gz", "ec_vocabulary.json", "mode.json", "te_tables.json.gz", "ranker.npz", "train.dmnd",
              "trembl_ev2.dmnd", "train_only.dmnd", "train_seqs_0.tsv.gz", "train_seqs_1.tsv.gz"]:
        assert (W / f).exists(), f
    assert (ROOT / "bin" / "diamond").exists()
    import gzip
    m = json.load(gzip.open(W / "labels.json.gz", "rt")); voc = set(json.load(open(W / "ec_vocabulary.json")))
    assert all(e in voc for l in m["labels"].values() for e in l)          # every label is a legal prediction
    cfg = Config.load(W / "mode.json")
    assert cfg.scoring == "vote" and cfg.vote_k == 50 and cfg.vote_power == 12 and cfg.source_weights["u2_"] == 0.1

def test_ranker_feature_width_matches():
    from ecpred.ranker import TreeRanker
    r = TreeRanker(ROOT / "weights" / "ranker.npz")
    assert int(r.feat.max()) <= len(FEATURES) + 5   # 27 search features + 6 reliability columns
