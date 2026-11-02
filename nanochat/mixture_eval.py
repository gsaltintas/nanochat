"""
Per-language bits-per-byte on held-out text and on the train probe (docs that are inside every
repetition tier, i.e. seen in training), always with canonical tokenization.

Each language's docs are packed as [BOS]+doc streams into consecutive windows (no cropping).
Windows are sharded across ranks and every rank runs the same number of steps.
"""
import pyarrow.parquet as pq
import torch

from nanochat.common import get_dist_info
from nanochat.loss_eval import evaluate_bpb


class LangEvaluator:
    def __init__(self, tokenizer, manifest, B, T, device, max_mb=None):
        self.B, self.T, self.device = B, T, device
        self.windows = {}  # (lang, split) -> LongTensor (n_windows, T+1)
        bos = tokenizer.get_bos_token_id()
        for lang, entry in manifest["languages"].items():
            for split in ("heldout", "probe"):
                t = pq.read_table(entry[split]["path"], columns=["text", "nbytes"])
                texts, nbytes = t.column("text").to_pylist(), t.column("nbytes").to_pylist()
                if max_mb is not None:
                    keep, tot = 0, 0
                    for nb in nbytes:
                        if tot >= max_mb * 1_000_000:
                            break
                        tot += nb
                        keep += 1
                    texts = texts[:keep]
                flat = []
                for ids in tokenizer.encode(texts):
                    flat.append(bos)
                    flat.extend(ids)
                n = (len(flat) - 1) // T
                flat = torch.tensor(flat[: n * T + 1], dtype=torch.long)
                # overlapping-by-one windows: row i covers flat[i*T : i*T + T + 1]
                idx = torch.arange(n).unsqueeze(1) * T + torch.arange(T + 1).unsqueeze(0)
                self.windows[(lang, split)] = flat[idx]

    @torch.no_grad()
    def evaluate(self, model, token_bytes):
        _, rank, _, world = get_dist_info()
        out = {}
        for (lang, split), w in self.windows.items():
            per_step = self.B * world
            steps = w.shape[0] // per_step
            if steps == 0:
                continue
            mine = w[: steps * per_step].view(steps, world, self.B, self.T + 1)[:, rank]

            def batches():
                for s in range(steps):
                    row = mine[s].to(self.device)
                    yield row[:, :-1].contiguous(), row[:, 1:].contiguous()

            out[f"{split}/bpb/{lang}"] = evaluate_bpb(model, batches(), steps, token_bytes)
        return out
