"""
Checkpoint bookkeeping for preemption-safe training (run:ai restarts an evicted pod with the *same command*).

A checkpoint step is usable only if it is *complete*: every rank has written its optimizer shard and per-rank
loader state, then rank 0 writes a COMPLETE_<step> marker after a barrier. A crash mid-save therefore leaves an
incomplete step that auto-resume ignores. After a new step is complete, older steps are pruned (keep last N).

Files per step (in the checkpoint dir):
    model_<step>.pt  meta_<step>.json  optim_<step>_rank<r>.pt  loader_<step>_rank<r>.json  COMPLETE_<step>
"""
import glob
import json
import os
import re


def _p(ckpt_dir, name):
    return os.path.join(ckpt_dir, name)


def latest_complete_step(ckpt_dir):
    steps = []
    for f in glob.glob(_p(ckpt_dir, "COMPLETE_*")):
        m = re.search(r"COMPLETE_(\d+)$", f)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else -1


def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def save_loader_state(ckpt_dir, step, rank, state):
    os.makedirs(ckpt_dir, exist_ok=True)
    _atomic_write(_p(ckpt_dir, f"loader_{step:06d}_rank{rank}.json"), json.dumps(state))


def load_loader_state(ckpt_dir, step, rank):
    path = _p(ckpt_dir, f"loader_{step:06d}_rank{rank}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mark_complete(ckpt_dir, step):
    _atomic_write(_p(ckpt_dir, f"COMPLETE_{step:06d}"), "ok\n")


def prune(ckpt_dir, keep=1):
    """Delete every checkpoint step older than the newest `keep` complete steps (keep <= 0: keep everything)."""
    if keep <= 0:
        return []
    steps = sorted({int(re.search(r"COMPLETE_(\d+)$", f).group(1)) for f in glob.glob(_p(ckpt_dir, "COMPLETE_*"))})
    doomed = steps[:-keep] if len(steps) > keep else []
    removed = []
    for s in doomed:
        for f in glob.glob(_p(ckpt_dir, f"*_{s:06d}*")) + [_p(ckpt_dir, f"COMPLETE_{s:06d}")]:
            if os.path.exists(f):
                os.remove(f)
                removed.append(os.path.basename(f))
    return removed
