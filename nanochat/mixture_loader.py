"""
Multilingual mixture dataloader with a fixed *data* (byte) budget.

Reads the frozen pools written by gul-epfl-tokenization/src/data/build_mixture.py
(manifest.json). For each language l with tier E_l:
  * the pool is the first tier_docs[E_l] documents of its (nested) E=1 pool, cycled
    repeatedly with a fresh seeded shuffle every epoch;
  * documents are drawn so that bytes drawn per language follow the target shares at all
    times (always draw from the language furthest behind its share, in bytes);
  * optionally, StochasTok is applied to chosen languages from a chosen epoch on, with an
    RNG seeded by (seed, doc_id, epoch) so every segmentation is reproducible;
  * packing never discards tokens: the token stream ([BOS] + doc, [BOS] + doc, ...) is cut
    into consecutive windows, so every token that is generated is trained on exactly once.
    (nanochat's default BOS-bestfit packing crops ~35% of tokens, which would make the
    canonical and stochastic arms see different amounts of unique text.)

Bytes consumed are counted when the first token of a document enters a batch.
"""
import json
import random
import zlib
from collections import deque

import numpy as np
import pyarrow.parquet as pq
import torch

from nanochat.common import get_dist_info
from nanochat.stochastok import StochasTok


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def parse_tiers(spec):
    """'rus_Cyrl=1,vie_Latn=4' -> {'rus_Cyrl': 1, 'vie_Latn': 4}"""
    if not spec:
        return {}
    return {k: int(v) for k, v in (kv.split("=") for kv in spec.split(","))}


def total_budget_bytes(manifest, tiers):
    return sum(e["tier_bytes"][str(tiers.get(l, 1))] * tiers.get(l, 1) for l, e in manifest["languages"].items())


class _Lang:
    def __init__(self, name, entry, tier, rank, world, seed):
        n = entry["tier_docs"][str(tier)]
        t = pq.read_table(entry["pool"]["path"], columns=["doc_id", "text", "nbytes"]).slice(0, n)
        self.name, self.tier, self.share = name, tier, entry["share"]
        self.doc_ids = t.column("doc_id").to_pylist()
        self.texts = t.column("text").to_pylist()
        self.nbytes = t.column("nbytes").to_pylist()
        self.n = n
        self.rank, self.world, self.seed = rank, world, seed
        self.epoch = 1
        self.pos = 0
        self.order = self._order(1)
        self.bytes_drawn = 0

    def _order(self, epoch):
        # crc32, not hash(): Python's str hash is randomized per process, ranks/runs must agree
        perm = np.random.RandomState(zlib.crc32(f"{self.seed}|{self.name}|{epoch}".encode())).permutation(self.n)
        return perm[self.rank::self.world]

    def draw(self):
        if self.pos >= len(self.order):
            self.epoch += 1
            self.pos = 0
            self.order = self._order(self.epoch)
        i = int(self.order[self.pos])
        self.pos += 1
        self.bytes_drawn += self.nbytes[i]
        return self.doc_ids[i], self.texts[i], self.nbytes[i], self.epoch


def mixture_data_loader(
    tokenizer, manifest, tiers, B, T, device="cuda", seed=0,
    stochastic_p=0.0, stochastic_langs=(), stochastic_from_epoch=1, char_safe=True,
    doclog=None, resume_state_dict=None, chunk_docs=64,
):
    ddp, rank, _, world = get_dist_info()
    langs = {
        name: _Lang(name, entry, tiers.get(name, 1), rank, world, seed)
        for name, entry in manifest["languages"].items()
    }
    # StochasTok is built (and the RNG streams used) only if some language needs it
    st = StochasTok(tokenizer, char_safe=char_safe) if stochastic_p > 0 and stochastic_langs else None
    bos = tokenizer.get_bos_token_id()
    if resume_state_dict:
        for name, s in resume_state_dict["langs"].items():
            L = langs[name]
            L.epoch, L.pos, L.bytes_drawn = s["epoch"], s["pos"], s["bytes_drawn"]
            L.order = L._order(L.epoch)

    buf = np.empty(0, dtype=np.int64)
    head = 0
    consumed_tokens = 0  # tokens that have left the stream (global index of buf[head])
    stream_end = 0       # global index one past the last token in buf
    pending = deque()    # (start_idx, meta) of docs whose first token has not been consumed yet
    bytes_consumed = {n: 0 for n in langs}
    tokens_consumed = {n: 0 for n in langs}
    batch_idx = resume_state_dict["batch_idx"] if resume_state_dict else 0
    use_cuda = device == "cuda"

    def refill():
        nonlocal buf, head, stream_end
        picked = []
        for _ in range(chunk_docs):
            name = min(langs, key=lambda n: langs[n].bytes_drawn / langs[n].share)
            doc_id, text, nbytes, epoch = langs[name].draw()
            picked.append((name, doc_id, text, nbytes, epoch))
        ids_list = tokenizer.encode([p[2] for p in picked])
        new = []
        for (name, doc_id, _, nbytes, epoch), ids in zip(picked, ids_list):
            stochastic = st is not None and name in stochastic_langs and epoch >= stochastic_from_epoch
            if stochastic:
                ids = st.expand(ids, stochastic_p, StochasTok.doc_rng(seed, doc_id, epoch))
            arr = np.empty(len(ids) + 1, dtype=np.int64)
            arr[0] = bos
            arr[1:] = ids
            h = zlib.crc32(arr.tobytes())
            pending.append((stream_end, (name, doc_id, epoch, nbytes, len(arr), stochastic, h)))
            stream_end += len(arr)
            new.append(arr)
        buf = np.concatenate([buf[head:]] + new)
        head = 0

    need = B * T + 1
    while True:
        while stream_end - consumed_tokens < need:
            refill()
        window = buf[head:head + need]
        x = torch.from_numpy(window[:-1].reshape(B, T).copy())
        y = torch.from_numpy(window[1:].reshape(B, T).copy())
        head += B * T
        consumed_tokens += B * T
        while pending and pending[0][0] < consumed_tokens:  # first token of this doc is now consumed
            _, (name, doc_id, epoch, nbytes, ntok, stochastic, h) = pending.popleft()
            bytes_consumed[name] += nbytes
            tokens_consumed[name] += ntok
            if doclog is not None:
                doclog.log(batch_idx, name, doc_id, epoch, nbytes, ntok, stochastic, h)
        state = {
            "batch_idx": batch_idx,
            "bytes_consumed": dict(bytes_consumed),
            "tokens_consumed": dict(tokens_consumed),
            "epochs": {n: L.epoch for n, L in langs.items()},
            # resume state is approximate: documents already drawn into the token buffer are skipped
            "langs": {n: {"epoch": L.epoch, "pos": L.pos, "bytes_drawn": L.bytes_drawn} for n, L in langs.items()},
        }
        batch_idx += 1
        yield x.to(device, non_blocking=use_cuda), y.to(device, non_blocking=use_cuda), state
