"""Rebuild the reference database used by predict.py: train.parquet + Swiss-Prot (external).

External resource (disclosed as the course requires):
  UniProtKB/Swiss-Prot, reviewed entries, fields accession / EC number / sequence, downloaded 31 Oct 2026 from
  https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&fields=accession,ec,sequence&query=reviewed:true
  (UniProt, CC BY 4.0). TrEMBL part: same fields with
  query=(reviewed:false) AND (ec:*) AND (existence:1); pass it with --trembl (prefix 'un_').

Steps:
  1. Keep Swiss-Prot entries with at least one EC in ec_vocabulary.json (other ECs, e.g. partial '3.2.2.-', dropped).
  2. Drop entries whose sequence is identical to a train.parquet sequence (train labels take precedence).
  3. Merge identical sequences (union of ECs). Each kept entry becomes a reference protein 'sp_<accession>'.
  4. Reference = train.parquet sequences + these entries; DIAMOND database built with `diamond makedb`.

To score without any external protein that is a test-split candidate, delete those records from
weights/swissprot_ext.fa.gz and rebuild:
    python tools/build_external_reference.py --train train.parquet --swissprot sprot.tsv.gz --exclude test_seqs.txt
    diamond makedb --in reference.fa -d weights/train.dmnd

Usage:
    python tools/build_external_reference.py --train train.parquet --swissprot sprot.tsv.gz [--exclude seqs.txt]
Writes reference.fa, weights/swissprot_ext.fa.gz (all external records) and weights/labels.json.gz.
"""
import argparse, gzip, json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--swissprot', required=True)
    ap.add_argument('--trembl', help='optional TrEMBL TSV (same columns), added after Swiss-Prot')
    ap.add_argument('--exclude', help='text file, one sequence per line, to leave out of the external part')
    a = ap.parse_args()
    voc = set(json.loads((ROOT / 'ec_vocabulary.json').read_text()))
    tr = pd.read_parquet(a.train)
    tr_seqs = tr.drop_duplicates('sequence')
    tr_lab = tr.groupby('sequence', sort=False).ec_number.apply(lambda s: list(dict.fromkeys(s))).to_dict()
    labels = {i: tr_lab[s] for i, s in zip(tr_seqs.id, tr_seqs.sequence)}
    freq = Counter(e for l in labels.values() for e in l)

    sp = pd.read_csv(a.swissprot, sep='\t', compression='infer')
    sp.columns = ['acc', 'ec', 'seq']
    sp['pre'] = 'sp_'
    if a.trembl:
        tb = pd.read_csv(a.trembl, sep='\t', compression='infer')
        tb.columns = ['acc', 'ec', 'seq']
        tb['pre'] = 'un_'
        sp = pd.concat([sp, tb], ignore_index=True)   # Swiss-Prot first: it wins on duplicate sequences
    sp = sp.dropna(subset=['ec'])
    sp['ecs'] = sp.ec.map(lambda x: [e.strip() for e in x.split(';') if e.strip() in voc])
    sp = sp[(sp.ecs.map(len) > 0) & ~sp.seq.isin(tr_lab)]
    if a.exclude:
        sp = sp[~sp.seq.isin(set(Path(a.exclude).read_text().split()))]
    g = sp.groupby('seq', sort=False).agg(acc=('acc', 'first'), pre=('pre', 'first'),
                                          ecs=('ecs', lambda s: list(dict.fromkeys(e for l in s for e in l))))
    print(f'train proteins: {len(labels)}  external proteins added: {len(g)}')

    with open('reference.fa', 'w') as f, gzip.open(ROOT / 'weights' / 'swissprot_ext.fa.gz', 'wt') as fx:
        for i, s in zip(tr_seqs.id, tr_seqs.sequence):
            f.write(f'>{i}\n{s}\n')
        for s, (acc, pre, ecs) in g.iterrows():
            rec = f'>{pre}{acc}\n{s}\n'
            f.write(rec)
            fx.write(rec)
            labels[f'{pre}{acc}'] = ecs
    with gzip.open(ROOT / 'weights' / 'labels.json.gz', 'wt') as f:
        json.dump({'labels': labels, 'freq': dict(freq)}, f)   # EC frequencies (tie-break) from train only


if __name__ == '__main__':
    main()
