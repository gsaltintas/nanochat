"""After a crash+resume run: the doc log must have no document consumed twice and batch indices must be consistent."""
import collections
import sqlite3
import sys

db = sqlite3.connect(f"{sys.argv[1]}/doclog_rank0.sqlite")
rows = db.execute("SELECT batch_idx, doc_id, epoch FROM doc_events ORDER BY rowid").fetchall()
dups = [k for k, n in collections.Counter((d, e) for _, d, e in rows).items() if n > 1]
decreasing = sum(1 for a, b in zip(rows, rows[1:]) if b[0] < a[0])
print(f"doc log: {len(rows)} rows, max batch {max(r[0] for r in rows)}, duplicate (doc, epoch): {len(dups)}, batch_idx decreases: {decreasing}")
assert not dups and decreasing == 0, "resume left duplicated or out-of-order rows"
print("CHECK OK")
