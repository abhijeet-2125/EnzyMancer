"""DIAMOND runner (original v16 logic): chunked queries, --bin spreading under RLIMIT_NOFILE, SIGXFSZ split-and-retry,
deadline watchdog.  New: on deadline the hits of already finished chunks are returned via DeadlineExceeded
instead of being thrown away."""
from __future__ import annotations
import os
import resource
import signal
import subprocess
import threading
import time
from pathlib import Path

MAX_CHUNK_BYTES = 12 * 1024 * 1024    # every file we write stays under 16 MiB
MAX_CHUNK_SEQS = 8000                 # bounds DIAMOND's temporary seed-hit files too
OUTFMT = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore",
          "qstart", "qend", "sstart", "send"]
CONFIGS = [(["-c1", "-b4"], 2), (["-c4", "-b1"], 1)]    # (extra args, threads) tried most efficient first


class DeadlineExceeded(RuntimeError):
    def __init__(self, msg, hits):
        super().__init__(msg)
        self.hits = hits


def parse_line(line):
    q, s, pid, length, ql, sl, ev, b, qs, qe, ss, se = line.rstrip("\n").split("\t")
    return q, (s, float(pid), int(ql), int(sl), float(ev), float(b), int(qs), int(qe), int(ss), int(se))


def write_chunks(items, workdir, tag):
    """items: iterable of (query index, sequence)."""
    paths, buf, size = [], [], 0

    def flush():
        nonlocal buf, size
        if buf:
            p = Path(workdir) / f"{tag}{len(paths)}.fa"
            p.write_text("".join(buf))
            paths.append(p)
            buf, size = [], 0
    for i, seq in items:
        rec = f">{i}\n{seq}\n"
        if size + len(rec) > MAX_CHUNK_BYTES or len(buf) >= MAX_CHUNK_SEQS:
            flush()
        buf.append(rec)
        size += len(rec)
    flush()
    return paths


def describe_rc(rc):
    if rc < 0:
        try:
            return f"killed by signal {signal.Signals(-rc).name}"
        except ValueError:
            return f"killed by signal {-rc}"
    return f"exit code {rc}"


def diamond_bins():
    try:
        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if soft == resource.RLIM_INFINITY:
            soft = 4096
    except Exception:
        soft = 256
    return int(max(16, min(128, soft - 64)))


def run_diamond(binary, db, chunks, mode, extra, threads, workdir, start, deadline_s, top_k=50, depth=0, bins=None):
    """Search each FASTA chunk against `db`; returns {query index: [hit,...]}."""
    bins = bins or diamond_bins()
    hits = {}
    for chunk in chunks:
        errf = Path(workdir) / "diamond.err"
        cmd = [binary, "blastp", "-q", str(chunk), "-d", str(db), "-k", str(top_k), "--threads", str(threads),
               "--quiet", "--ignore-warnings", *extra, "--bin", str(bins), "--tmpdir", str(workdir), f"--{mode}",
               "--outfmt", "6", *OUTFMT]
        local = {}
        with errf.open("w") as err:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
            watchdog = threading.Timer(max(1.0, deadline_s - (time.time() - start)), proc.kill)
            watchdog.start()
            for line in proc.stdout:
                q, h = parse_line(line)
                local.setdefault(int(q), []).append(h)
            rc = proc.wait()
            watchdog.cancel()
        if time.time() - start >= deadline_s - 1 and rc != 0:
            raise DeadlineExceeded(f"diamond {mode} stopped by watchdog at {deadline_s:.0f}s", hits)
        if rc == -signal.SIGXFSZ and depth < 3:
            recs = chunk.read_text().split(">")[1:]
            if len(recs) > 1:
                parts = []
                for k in range(4):
                    sub = recs[k::4]
                    if sub:
                        p = chunk.with_name(f"{chunk.stem}_{depth}_{k}.fa")
                        p.write_text("".join(">" + r for r in sub))
                        parts.append(p)
                chunk.unlink()
                hits.update(run_diamond(binary, db, parts, mode, extra, threads, workdir, start, deadline_s,
                                        top_k, depth + 1, bins))
                continue
        msg = errf.read_text()[-2000:].strip() if (rc != 0 and errf.exists()) else ""
        if rc != 0 and "Too many open files" in msg and bins > 16:
            hits.update(run_diamond(binary, db, [chunk], mode, extra, threads, workdir, start, deadline_s,
                                    top_k, depth, bins // 2))
            continue
        if rc != 0:
            raise RuntimeError(f"diamond {mode} {extra} t={threads}: {describe_rc(rc)}; stderr: {msg or '(empty)'}")
        hits.update(local)
        chunk.unlink()
    return hits
