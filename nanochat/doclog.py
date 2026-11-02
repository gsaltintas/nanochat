"""
Per-run, per-rank SQLite log of every document sampled by the training dataloader.

One row per (document, sampling event): which language/document, which epoch it was on,
which micro-batch it entered, how many bytes/tokens it had, whether stochastic
tokenization was applied, and a hash of the exact token sequence used. Segmentations are
reproducible from (seed, doc_id, epoch), so token ids are not stored.

Each rank writes its own file (<dir>/doclog_rank{r}.sqlite) to avoid write contention;
merge afterwards with `ATTACH` or pandas.
"""
import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_events (
    run_id TEXT, rank INTEGER, batch_idx INTEGER, lang TEXT, doc_id TEXT, epoch INTEGER,
    n_bytes INTEGER, n_tokens INTEGER, stochastic INTEGER, seg_hash INTEGER
);
CREATE INDEX IF NOT EXISTS idx_doc ON doc_events (doc_id);
"""


class DocLog:
    def __init__(self, log_dir, run_id, rank, flush_every=20000, fresh=True):
        """fresh=True starts an empty log (a re-run of the same tag must not append to the old run's rows);
        pass fresh=False when resuming a run."""
        os.makedirs(log_dir, exist_ok=True)
        self.run_id, self.rank, self.flush_every = run_id, rank, flush_every
        self.path = os.path.join(log_dir, f"doclog_rank{rank}.sqlite")
        if fresh:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(self.path + suffix):
                    os.remove(self.path + suffix)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.buf = []

    def log(self, batch_idx, lang, doc_id, epoch, n_bytes, n_tokens, stochastic, seg_hash):
        self.buf.append((self.run_id, self.rank, batch_idx, lang, doc_id, epoch, n_bytes, n_tokens, int(stochastic), seg_hash))
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if self.buf:
            self.db.executemany("INSERT INTO doc_events VALUES (?,?,?,?,?,?,?,?,?,?)", self.buf)
            self.db.commit()
            self.buf = []

    def truncate_after(self, batch_idx):
        """Drop events at or after batch_idx (used when resuming from a checkpoint)."""
        self.db.execute("DELETE FROM doc_events WHERE batch_idx >= ?", (batch_idx,))
        self.db.commit()

    def close(self):
        self.flush()
        self.db.close()
