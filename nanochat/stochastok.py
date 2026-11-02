"""
StochasTok (Sims et al., 2025): post-tokenization stochastic expansion.

Repeatedly pick a random position in the (evolving) token sequence and, if the token has a
valid binary split (both halves are vocab entries), replace it with a random split.
`max(1, int(len * p))` steps are taken per sequence, so ~p of the tokens are touched.

Differences from the research implementation in src/tokenization/stochastok.py:
  * takes a local random.Random (seeded per document and epoch) instead of the global RNG
  * `char_safe=True` only allows splits whose two halves are both valid UTF-8, i.e. splits
    that fall on character boundaries (on byte-level vocabs, a plain split can cut through a
    multi-byte character and produce byte-fragment tokens)
"""
import random


def _utf8_ok(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


class StochasTok:
    def __init__(self, hf_tokenizer, char_safe=True):
        from nanochat.hf_tokenizer import _gpt2_byte_decoder
        tok = hf_tokenizer.tok
        added = {int(i) for i in tok.get_added_tokens_decoder()}
        dec = _gpt2_byte_decoder()
        pieces = {}
        for s, i in tok.get_vocab(with_added_tokens=True).items():
            if i in added:
                continue
            if hf_tokenizer.byte_level:
                try:
                    pieces[i] = bytes(dec[c] for c in s)
                except KeyError:
                    continue
            else:
                pieces[i] = s.replace("▁", " ")
        rev = {p: i for i, p in pieces.items()}
        self.char_safe = char_safe
        self.pairs = {}
        for i, p in pieces.items():
            for k in range(1, len(p)):
                a, b = p[:k], p[k:]
                if a in rev and b in rev:
                    if char_safe and hf_tokenizer.byte_level and not (_utf8_ok(a) and _utf8_ok(b)):
                        continue
                    self.pairs.setdefault(i, []).append((rev[a], rev[b]))

    def expand(self, ids, p, rng):
        out = list(ids)
        if not out or p <= 0:
            return out
        pairs = self.pairs
        for _ in range(max(1, int(len(out) * p))):
            i = rng.randrange(len(out))
            sp = pairs.get(out[i])
            if sp:
                a, b = sp[rng.randrange(len(sp))] if len(sp) > 1 else sp[0]
                out[i:i + 1] = (a, b)
        return out

    @staticmethod
    def doc_rng(seed, doc_id, epoch):
        return random.Random(f"{seed}|{doc_id}|{epoch}")
