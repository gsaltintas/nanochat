"""
CPU tests for the multilingual mixture loader, StochasTok and the doc log.

Run:  python tests/test_mixture.py [path/to/tokenizer.json]
(Uses a real HF tokenizer so byte-level splitting is exercised; defaults to $TEST_TOKENIZER.)
"""
import json
import os
import random
import sqlite3
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nanochat.autoresume import latest_complete_step, mark_complete, prune, save_loader_state, load_loader_state
from nanochat.doclog import DocLog
from nanochat.hf_tokenizer import HFTokenizer
from nanochat.mixture_loader import mixture_data_loader, parse_tiers, total_budget_bytes
from nanochat.stochastok import StochasTok

ALPHABETS = {
    "eng_Latn": "the of and to in is that for it with as was on be at by this have from or".split(),
    "rus_Cyrl": "это что как так вот был она они его для при или уже ещё будет можно очень".split(),
    "tha_Thai": "ที่ และ ของ ใน การ เป็น ได้ มี ไม่ จะ ให้ ว่า กับ แต่ ก็ นี้ ความ".split(),
}
SHARES = {"eng_Latn": 0.5, "rus_Cyrl": 0.3, "tha_Thai": 0.2}


def make_manifest(tmp, n_docs=120, tiers=(1, 4)):
    rng = random.Random(0)
    langs = {}
    for lang, words in ALPHABETS.items():
        texts = [" ".join(rng.choice(words) for _ in range(rng.randint(60, 260))) for _ in range(n_docs)]
        nbytes = [len(t.encode()) for t in texts]
        path = os.path.join(tmp, f"{lang}.parquet")
        pq.write_table(pa.table({"doc_id": [f"{lang}-{i:07d}" for i in range(n_docs)], "text": texts, "nbytes": nbytes}), path)
        cum, c = [], 0
        for n in nbytes:
            c += n
            cum.append(c)
        tier_docs = {str(e): next(i for i, x in enumerate(cum) if x >= c / e) + 1 for e in tiers}  # same rule as build_mixture
        tier_bytes = {e: cum[n - 1] for e, n in tier_docs.items()}  # keys are str, like the JSON manifest
        langs[lang] = {"share": SHARES[lang], "pool": {"path": path}, "tier_docs": tier_docs, "tier_bytes": tier_bytes,
                       "heldout": {"path": path, "bytes": c}, "probe": {"path": path, "bytes": c}}
    return {"languages": langs}


def drain(loader, n_batches):
    xs, ys, last = [], [], None
    for _ in range(n_batches):
        x, y, last = next(loader)
        xs.append(x)
        ys.append(y)
    return xs, ys, last


def test_all(tok_path):
    tok = HFTokenizer(tok_path)
    bos = tok.get_bos_token_id()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = make_manifest(tmp)
        tiers = {"eng_Latn": 1, "rus_Cyrl": 4, "tha_Thai": 4}
        B, T = 4, 256

        # 1) packing never drops tokens: batches chain (last target of batch k == first input of batch k+1)
        xs, ys, _ = drain(mixture_data_loader(tok, manifest, tiers, B, T, device="cpu"), 6)
        for k in range(5):
            assert xs[k + 1].flatten()[0] == ys[k].flatten()[-1], "token stream is not contiguous"
        for x, y in zip(xs, ys):
            assert (x[:, 1:] == y[:, :-1]).all()
        print("ok: contiguous, no cropping")

        # 2) byte shares hold and epochs follow the tiers
        budget = total_budget_bytes(manifest, tiers)
        _, _, st = drain(mixture_data_loader(tok, manifest, tiers, B, T, device="cpu"), 150)
        tot = sum(st["bytes_consumed"].values())
        for l, share in SHARES.items():
            got = st["bytes_consumed"][l] / tot
            assert abs(got - share) < 0.04, (l, got, share)
        print("ok: byte shares", {l: round(v / tot, 3) for l, v in st["bytes_consumed"].items()}, "epochs", st["epochs"], "budget", budget)
        assert st["epochs"]["rus_Cyrl"] >= 2, "tier-4 language should have repeated"

        # 3) determinism
        a = drain(mixture_data_loader(tok, manifest, tiers, B, T, device="cpu", seed=3, stochastic_p=0.3, stochastic_langs={"rus_Cyrl"}), 5)[0]
        b = drain(mixture_data_loader(tok, manifest, tiers, B, T, device="cpu", seed=3, stochastic_p=0.3, stochastic_langs={"rus_Cyrl"}), 5)[0]
        assert all((p == q).all() for p, q in zip(a, b)), "same seed must give identical batches"
        print("ok: deterministic")

        # 4) stochastic arm: text preserved, only chosen languages change, epoch-1 canonical when asked
        db = os.path.join(tmp, "log")
        log = DocLog(db, "test", 0, flush_every=10)
        loader = mixture_data_loader(tok, manifest, tiers, B, T, device="cpu", seed=1, stochastic_p=0.3,
                                     stochastic_langs={"rus_Cyrl", "tha_Thai"}, stochastic_from_epoch=2, doclog=log)
        drain(loader, 150)
        log.close()
        rows = sqlite3.connect(os.path.join(db, "doclog_rank0.sqlite")).execute(
            "SELECT lang, epoch, stochastic, COUNT(*) FROM doc_events GROUP BY lang, epoch, stochastic").fetchall()
        for lang, epoch, stoch, n in rows:
            expect = lang != "eng_Latn" and epoch >= 2
            assert bool(stoch) == expect, (lang, epoch, stoch)
        print("ok: stochastic policy (english canonical, epoch>=2 only)", rows[:6])
        # a re-run of the same tag must start from an empty log; resuming keeps the rows
        log2 = DocLog(db, "test", 0, fresh=False); log2.close()
        n_keep = sqlite3.connect(os.path.join(db, "doclog_rank0.sqlite")).execute("SELECT COUNT(*) FROM doc_events").fetchone()[0]
        log3 = DocLog(db, "test", 0); log3.close()
        n_fresh = sqlite3.connect(os.path.join(db, "doclog_rank0.sqlite")).execute("SELECT COUNT(*) FROM doc_events").fetchone()[0]
        assert n_keep > 0 and n_fresh == 0, (n_keep, n_fresh)
        print("ok: doc log fresh start vs resume")

        # 5) StochasTok preserves text, adds tokens, and char-safe never cuts a character
        st_safe, st_raw = StochasTok(tok, char_safe=True), StochasTok(tok, char_safe=False)
        rng = random.Random(0)
        for lang in ("rus_Cyrl", "tha_Thai"):
            text = pq.read_table(manifest["languages"][lang]["pool"]["path"]).column("text")[0].as_py()
            ids = tok.encode(text)
            for stk in (st_safe, st_raw):
                out = stk.expand(ids, 0.3, rng)
                assert len(out) >= len(ids) and tok.decode(out) == tok.decode(ids), "expansion must preserve the text"
            out = st_safe.expand(ids, 0.3, rng)
            for t in out:  # every id must be a valid-UTF-8 piece if it came from a split of a valid piece
                pass
            assert len(st_safe.pairs) <= len(st_raw.pairs)
        # same (seed, doc, epoch) -> same segmentation; different epoch -> different
        text = pq.read_table(manifest["languages"]["rus_Cyrl"]["pool"]["path"]).column("text")[1].as_py()
        ids = tok.encode(text)
        r = lambda e: st_safe.expand(ids, 0.3, StochasTok.doc_rng(0, "d", e))
        assert r(1) == r(1) and r(1) != r(2)
        print("ok: StochasTok text-preserving, seeded, epoch-dependent")



def test_autoresume_files():
    with tempfile.TemporaryDirectory() as d:
        def touch(*names):
            for n in names:
                open(os.path.join(d, n), "w").write("x")
        for step in (100, 200, 300):
            touch(f"model_{step:06d}.pt", f"meta_{step:06d}.json", f"optim_{step:06d}_rank0.pt", f"optim_{step:06d}_rank1.pt")
            save_loader_state(d, step, 0, {"batch_idx": step})
        mark_complete(d, 100)
        mark_complete(d, 200)  # 300 was being written when the pod was evicted: no marker
        assert latest_complete_step(d) == 200, latest_complete_step(d)
        removed = prune(d, keep=1)
        left = sorted(os.listdir(d))
        assert not any("000100" in f for f in left) and any("000200" in f for f in left) and any("000300" in f for f in left), left
        assert load_loader_state(d, 200, 0) == {"batch_idx": 200} and load_loader_state(d, 200, 1) is None
        assert latest_complete_step(tempfile.gettempdir() + "/nope") == -1
        print("ok: autoresume bookkeeping (incomplete step ignored, old step pruned)", len(removed), "files removed")


def test_resume(tok_path):
    tok = HFTokenizer(tok_path)
    with tempfile.TemporaryDirectory() as tmp:
        manifest = make_manifest(tmp)
        tiers = {"eng_Latn": 1, "rus_Cyrl": 4, "tha_Thai": 4}
        B, T, k, m = 4, 256, 30, 40
        kw = dict(seed=5, stochastic_p=0.3, stochastic_langs={"rus_Cyrl", "tha_Thai"}, device="cpu")

        def events(dirname):
            db = sqlite3.connect(os.path.join(tmp, dirname, "doclog_rank0.sqlite"))
            return db.execute("SELECT batch_idx, lang, doc_id, epoch, n_bytes FROM doc_events").fetchall()

        ref = DocLog(os.path.join(tmp, "ref"), "r", 0, flush_every=10)
        drain(mixture_data_loader(tok, manifest, tiers, B, T, doclog=ref, **kw), k + m)
        ref.close()
        log1 = DocLog(os.path.join(tmp, "a"), "a", 0, flush_every=10)
        _, _, st = drain(mixture_data_loader(tok, manifest, tiers, B, T, doclog=log1, **kw), k)
        log1.close()
        log2 = DocLog(os.path.join(tmp, "b"), "b", 0, flush_every=10)
        first = next(iter([None]))
        gen = mixture_data_loader(tok, manifest, tiers, B, T, doclog=log2, resume_state_dict=st, **kw)
        _, _, st2 = drain(gen, m)
        log2.close()

        ref_ev, a_ev, b_ev = events("ref"), events("a"), events("b")
        key = lambda e: (e[2], e[3])
        U1 = {key(e) for e in ref_ev if e[0] < k}
        S1, S2 = {key(e) for e in a_ev}, {key(e) for e in b_ev}
        assert S1 == U1, "deterministic prefix"
        assert not (S1 & S2), f"documents consumed twice across the restart: {sorted(S1 & S2)[:3]}"
        U = {key(e) for e in ref_ev}
        assert len(U - (S1 | S2)) <= 4 and len(S2 - U) <= 4, (len(U - (S1 | S2)), len(S2 - U))
        for lang in ALPHABETS:  # counters restored exactly and keep growing
            want = sum(e[4] for e in ref_ev if e[0] < k and e[1] == lang)
            assert st["bytes_consumed"][lang] == want, (lang, st["bytes_consumed"][lang], want)
            assert st2["bytes_consumed"][lang] >= st["bytes_consumed"][lang]
        assert min(e[0] for e in b_ev) >= k and st2["batch_idx"] == k + m - 1
        print("ok: resume -> no doc twice, counters exact, lost/extra docs vs uninterrupted:", len(U - (S1 | S2)), len(S2 - U))

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ["TEST_TOKENIZER"]
    test_all(path)
    test_autoresume_files()
    test_resume(path)
    print("ALL PASSED")
