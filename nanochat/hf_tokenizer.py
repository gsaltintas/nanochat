"""
Thin wrapper that lets nanochat use a HuggingFace `tokenizers` BPE (tokenizer.json)
through the same interface as RustBPETokenizer (the parts the training code uses).

Needed because the multilingual experiments use pre-trained HF tokenizers
(byte-level BPE with a GPT-2-style byte->unicode alphabet).
"""
import json
import os

from tokenizers import Tokenizer


def _gpt2_byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class HFTokenizer:
    def __init__(self, path, bos_token="<s>"):
        path = os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path
        self.path = path
        self.tok = Tokenizer.from_file(path)
        with open(path) as f:
            self.spec = json.load(f)
        self.bos_token = bos_token
        self.bos_token_id = self.tok.token_to_id(bos_token)
        assert self.bos_token_id is not None, f"{bos_token} not in tokenizer vocab"
        self.byte_level = (self.spec.get("decoder") or {}).get("type") == "ByteLevel"
        vocab = self.tok.get_vocab(with_added_tokens=True)
        self._vocab_size = max(vocab.values()) + 1

    def get_vocab_size(self):
        return self._vocab_size

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode_special(self, text):
        if text == "<|bos|>":  # nanochat's name for the document delimiter
            return self.bos_token_id
        tid = self.tok.token_to_id(text)
        assert tid is not None, f"unknown special token {text}"
        return tid

    def encode(self, text, prepend=None, append=None, num_threads=8):
        """Canonical encoding of a str or list[str] (no special tokens added by the tokenizer)."""
        pre = None if prepend is None else (prepend if isinstance(prepend, int) else self.encode_special(prepend))
        app = None if append is None else (append if isinstance(append, int) else self.encode_special(append))
        single = isinstance(text, str)
        encs = self.tok.encode_batch([text] if single else text, add_special_tokens=False)
        out = []
        for e in encs:
            ids = list(e.ids)
            if pre is not None:
                ids.insert(0, pre)
            if app is not None:
                ids.append(app)
            out.append(ids)
        return out[0] if single else out

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def token_bytes_list(self):
        """UTF-8 byte length per token id; 0 for added/special tokens (excluded from bpb)."""
        added = {int(i) for i in self.tok.get_added_tokens_decoder()}
        out = [0] * self._vocab_size
        dec = _gpt2_byte_decoder()
        for s, i in self.tok.get_vocab(with_added_tokens=True).items():
            if i in added:
                continue
            if self.byte_level:
                try:
                    out[i] = len(bytes(dec[c] for c in s))
                except KeyError:
                    out[i] = len(s.encode("utf-8"))
            else:
                out[i] = len(s.replace("▁", " ").encode("utf-8"))
        return out

    def token_bytes(self, device="cpu"):
        import torch
        return torch.tensor(self.token_bytes_list(), dtype=torch.int32, device=device)
