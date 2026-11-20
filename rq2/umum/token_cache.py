"""token_cache.py - tokenisasi seluruh esai sekali saja (dipotong pada max_len) dan simpan id-nya.

Satu berkas per keluarga model: work/tokens_deberta.npz dan work/tokens_bert.npz. Urutan baris = row_id data resmi.
Pemakaian:  python umum/token_cache.py deberta   |   python umum/token_cache.py bert
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402

NAMA_MODEL = {"deberta": "model_deberta", "bert": "model_bert"}


def berkas(kunci):
    return C.wpath("work", f"tokens_{kunci}.npz")


def buat(df, kunci, cfg=None, tulis=True):
    from transformers import AutoTokenizer
    cfg = cfg or C.config()
    nama = cfg[NAMA_MODEL[kunci]]
    texts = [str(t).strip() for t in df.Text.tolist()]
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(nama)
    except Exception as e:                                   # tokenizer cepat dapat gagal tanpa sentencepiece/protobuf
        print("tokenizer cepat gagal (", repr(e)[:120], ") -> memakai tokenizer lambat")
        tok = AutoTokenizer.from_pretrained(nama, use_fast=False)
    L = cfg["max_len"]
    enc = tok(texts, truncation=True, max_length=L)["input_ids"]
    full = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
    lens_full = np.array([len(x) for x in full]) + 2
    flat = np.concatenate([np.asarray(x, dtype=np.int32) for x in enc])
    offs = np.concatenate([[0], np.cumsum([len(x) for x in enc])]).astype(np.int64)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    if tulis:
        os.makedirs(os.path.dirname(berkas(kunci)), exist_ok=True)
        np.savez(berkas(kunci), flat=flat, offs=offs, pad_id=pad, max_len=L, full_len=lens_full)
    print(f"[{kunci}] {len(enc)} esai; panjang token tanpa pemotongan: median {np.median(lens_full):.0f}, "
          f"persentil 95 {np.percentile(lens_full, 95):.0f}, maks {lens_full.max()}")
    print(f"[{kunci}] dipotong pada {L} token: {100 * (lens_full > L).mean():.1f}% esai (catat di keterbatasan); {time.time() - t0:.0f} detik")
    return flat, offs, int(pad)


def muat(kunci):
    fn = berkas(kunci)
    if not os.path.exists(fn):
        raise SystemExit(f"{fn} belum ada. Jalankan:  python umum/token_cache.py {kunci}")
    z = np.load(fn)
    return z["flat"], z["offs"], int(z["pad_id"])


if __name__ == "__main__":
    kunci = sys.argv[1] if len(sys.argv) > 1 else "deberta"
    if kunci not in NAMA_MODEL:
        raise SystemExit("pilih: deberta atau bert")
    buat(C.muat_baris(), kunci)
