"""Pure numpy/scipy fallback: 3-mer TF-IDF cosine nearest neighbours.

Used only when DIAMOND cannot run in the evaluation sandbox.
"""
import numpy as np
from scipy import sparse

AA = 'ACDEFGHIKLMNPQRSTVWY'
LUT = np.full(256, 20, dtype=np.int64)
for i, a in enumerate(AA):
    LUT[ord(a)] = i
K = 3
DIM = 21 ** K


def kmer_matrix(seqs, chunk=10000):
    """Rows = sequences, columns = 3-mer ids (counts). Built in chunks to bound memory."""
    parts = [_kmer_block(seqs[i:i + chunk]) for i in range(0, len(seqs), chunk)]
    return sparse.vstack(parts, format='csr') if parts else sparse.csr_matrix((0, DIM), dtype=np.float32)


def _kmer_block(seqs):
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    codes = LUT[np.frombuffer(''.join(seqs).encode('latin-1'), dtype=np.uint8)]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    km = codes[:-2] * 441 + codes[1:-1] * 21 + codes[2:]
    pos = np.arange(len(km))
    row = np.repeat(np.arange(len(seqs)), lens)[:len(km)]
    valid = np.ones(len(km), dtype=bool)
    ends = starts + lens
    # drop k-mers that cross a sequence boundary
    for off in (1, 2):
        b = ends - off
        b = b[(b >= 0) & (b < len(km))]
        valid[b] = False
    valid &= (codes[:-2] < 20) & (codes[1:-1] < 20) & (codes[2:] < 20)
    m = sparse.csr_matrix((np.ones(valid.sum(), dtype=np.float32), (row[valid], km[valid])),
                          shape=(len(seqs), DIM))
    m.sum_duplicates()
    return m


class KmerIndex:
    def __init__(self, train_seqs):
        X = kmer_matrix(train_seqs)
        df = np.bincount(X.indices, minlength=DIM)
        self.idf = np.log((1 + X.shape[0]) / (1 + df)).astype(np.float32) + 1
        self.X = self._norm(X)
        self.XT = self.X.T.tocsr()

    def _norm(self, X):
        X = X.multiply(self.idf).tocsr()
        X.data = np.sqrt(X.data)
        n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-9
        return sparse.diags(1 / n) @ X

    def query_fast(self, seqs, k=10, shortlist=50, dim=384, batch=256):
        """Random-projection shortlist, then exact sparse cosine on the shortlist."""
        R = np.random.default_rng(0).standard_normal((DIM, dim)).astype(np.float32)
        P = np.asarray(self.X @ R, dtype=np.float32)
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
        Q = self._norm(kmer_matrix(seqs))
        QP = np.asarray(Q @ R, dtype=np.float32)
        QP /= np.linalg.norm(QP, axis=1, keepdims=True) + 1e-9
        out_i, out_s = [], []
        for b in range(0, Q.shape[0], batch):
            s = QP[b:b + batch] @ P.T
            cand = np.argpartition(-s, shortlist, axis=1)[:, :shortlist]
            qb = Q[b:b + batch]
            exact = np.zeros(cand.shape, dtype=np.float32)
            for r in range(cand.shape[0]):
                exact[r] = (self.X[cand[r]] @ qb[r].T).toarray().ravel()
            o = np.argsort(-exact, axis=1)[:, :k]
            out_i.append(np.take_along_axis(cand, o, 1))
            out_s.append(np.take_along_axis(exact, o, 1))
        return np.vstack(out_i), np.vstack(out_s)

    def query(self, seqs, k=10, batch=512):
        Q = self._norm(kmer_matrix(seqs))
        out_i, out_s = [], []
        for b in range(0, Q.shape[0], batch):
            S = (Q[b:b + batch] @ self.XT).toarray()
            idx = np.argpartition(-S, k, axis=1)[:, :k]
            sc = np.take_along_axis(S, idx, 1)
            o = np.argsort(-sc, axis=1)
            out_i.append(np.take_along_axis(idx, o, 1))
            out_s.append(np.take_along_axis(sc, o, 1))
        return np.vstack(out_i), np.vstack(out_s)
