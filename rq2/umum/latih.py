"""latih.py - pelatihan jaringan saraf yang dipakai bersama oleh K2-K6 (kepala kelas) dan B2-B3 (kepala regresi).

Dua jenis kepala (tidak ada yang lain):
  kelas   : pooling -> dropout -> Linear(h, 5 logit); galat = cross-entropy terhadap sebaran target (label lunak);
            prediksi = nilai harapan sebaran (1..5).                                   [K2-K6]
  regresi : pooling -> dropout -> Linear(h, 1) (+ sigmoid bila diminta); galat = MSE terhadap skor yang dinormalkan ke [0,1];
            prediksi = 1 + 4 * keluaran.                                               [B2, B3]

Resep (config.json, tanpa penyetelan per metode, tanpa penghentian dini, epoch terakhir dipakai):
  AdamW, lr encoder dan lr kepala terpisah, weight decay tidak dikenakan pada bias/LayerNorm, pemanasan linear lalu peluruhan linear,
  batch efektif = micro-batch x akumulasi, bf16 lewat autocast dengan bobot fp32, pemotongan gradien.
Urutan esai tiap epoch dan inisialisasi kepala ditentukan oleh (seed, lipatan) SAJA, jadi sama untuk semua metode
(common random numbers): perbedaan antar metode berasal dari target, bukan dari kebisingan acak.
"""
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402


class Skor(nn.Module):
    def __init__(self, nama_enc, jenis, pooling, sigmoid, dropout, grad_ckpt):
        super().__init__()
        self.enc = C.muat_encoder(nama_enc)
        if grad_ckpt:
            self.enc.gradient_checkpointing_enable()
        self.jenis, self.pooling, self.sigmoid = jenis, pooling, sigmoid
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.enc.config.hidden_size, 5 if jenis == "kelas" else 1)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "cls":
            pooled = h[:, 0]
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        out = self.head(self.drop(pooled)).float()
        if self.jenis == "regresi":
            out = out.squeeze(-1)
            if self.sigmoid:
                out = torch.sigmoid(out)
        return out


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def collate(flat, offs, pad, idx):
    seqs = [flat[offs[i]:offs[i + 1]] for i in idx]
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), pad, dtype=np.int64)
    mask = np.zeros((len(seqs), L), dtype=np.int64)
    for k, s in enumerate(seqs):
        ids[k, :len(s)] = s
        mask[k, :len(s)] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


def soft_ce(logits, target):
    return -(target * torch.log_softmax(logits, -1)).sum(-1).mean()


def skor_ke_norm(y):
    """Skor 1..5 -> [0,1] (Fan & Yun 2026: normalisasi ke [0,1]; Uto & Okano 2021: sigmoid pada [0,1])."""
    return (np.asarray(y, float) - 1.0) / 4.0


def norm_ke_skor(o):
    return 1.0 + 4.0 * np.asarray(o, float)


@torch.no_grad()
def prediksi(model, flat, offs, pad, idx, eval_batch, dev, amp):
    model.eval()
    urut = np.argsort([-(offs[i + 1] - offs[i]) for i in idx], kind="stable")      # panjang dulu -> memori stabil
    k_out = 5 if model.jenis == "kelas" else None
    out = np.zeros((len(idx), 5) if k_out else (len(idx),), dtype=np.float32)
    for s in range(0, len(idx), eval_batch):
        sel = urut[s:s + eval_batch]
        ids, mask = collate(flat, offs, pad, [idx[j] for j in sel])
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            o = model(ids.to(dev), mask.to(dev))
        out[sel] = (torch.softmax(o, -1) if k_out else o).cpu().numpy()
    return out


def latih_prediksi(tok, nama_enc, jenis, pooling, sigmoid, idx_latih, target, idx_uji, seed, fold, resep, opsi, label):
    """tok = (flat, offs, pad). target: P (n,5) bila jenis='kelas', skor 1..5 (n,) bila 'regresi'.
    resep: dict epochs, micro_batch, accum, lr_backbone, lr_head, weight_decay, warmup_ratio, dropout, grad_clip, bf16, grad_ckpt, eval_batch.
    opsi: objek dengan smoke, cpu, no_grad_ckpt, micro_batch (0 = ikuti resep), ukur, bench_steps.
    Mengembalikan dict(pred, probs atau None, log)."""
    flat, offs, pad = tok
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu" and not (opsi.smoke or opsi.cpu):
        raise SystemExit("GPU tidak terdeteksi. Jalankan check_env.py (atau pakai --cpu hanya untuk uji kecil).")
    amp = bool(resep["bf16"]) and dev.type == "cuda"
    idx_latih, idx_uji = np.asarray(idx_latih), np.asarray(idx_uji)
    y = np.asarray(target, np.float32) if jenis == "kelas" else skor_ke_norm(target).astype(np.float32)
    if opsi.smoke:
        idx_latih, y, idx_uji = idx_latih[:48], y[:48], idx_uji[:32]

    set_seed(seed * 1000 + fold)                                     # inisialisasi kepala dan dropout sama untuk semua metode
    model = Skor(nama_enc, jenis, pooling, sigmoid, resep["dropout"], resep["grad_ckpt"] and not opsi.no_grad_ckpt).to(dev)
    tdk_peluruhan = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    enc = list(model.enc.named_parameters())
    grup = [
        {"params": [p for n, p in enc if not any(k in n for k in tdk_peluruhan)], "lr": resep["lr_backbone"], "weight_decay": resep["weight_decay"]},
        {"params": [p for n, p in enc if any(k in n for k in tdk_peluruhan)], "lr": resep["lr_backbone"], "weight_decay": 0.0},
        {"params": list(model.head.parameters()), "lr": resep["lr_head"], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(grup, betas=(0.9, 0.999), eps=1e-6)
    eff = resep["micro_batch"] * resep["accum"]
    mb = opsi.micro_batch or resep["micro_batch"]
    accum = max(1, eff // mb)                                        # batch efektif tetap
    epochs = 1 if opsi.smoke else resep["epochs"]
    n = len(idx_latih)
    langkah_per_epoch = math.ceil(math.ceil(n / mb) / accum)
    total = langkah_per_epoch * epochs
    sched = get_linear_schedule_with_warmup(opt, int(resep["warmup_ratio"] * total), total)
    rng = np.random.default_rng(seed * 100003 + fold * 17)           # urutan esai sama untuk semua metode
    log = dict(label=label, fold=fold, seed=seed, n_latih=int(n), n_uji=int(len(idx_uji)), micro_batch=mb, accum=accum,
               batch_efektif=int(mb * accum), epochs=epochs, jenis=jenis, pooling=pooling, sigmoid=bool(sigmoid), epoch_loss=[])
    t0 = time.time()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_bench = None
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        perm = rng.permutation(n)
        tot, cnt, micro = 0.0, 0, 0
        for s in range(0, n, mb):
            b = perm[s:s + mb]
            ids, mask = collate(flat, offs, pad, idx_latih[b])
            tgt = torch.from_numpy(y[b]).to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                out = model(ids.to(dev), mask.to(dev))
            loss = soft_ce(out, tgt) if jenis == "kelas" else torch.mean((out - tgt) ** 2)
            (loss / accum).backward()
            tot += float(loss) * len(b)
            cnt += len(b)
            micro += 1
            if micro % accum == 0 or s + mb >= n:
                torch.nn.utils.clip_grad_norm_(model.parameters(), resep["grad_clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if opsi.ukur:
                if micro == 3:
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t_bench = time.time()
                if micro == 3 + opsi.bench_steps:
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    sps = opsi.bench_steps * mb / (time.time() - t_bench)
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
                    menit = (n * resep["epochs"] / sps + len(idx_uji) / (3 * sps)) / 60
                    print(f"\n[ukur] {sps:.1f} esai/detik (latih), memori puncak {peak:.2f} GB")
                    print(f"[ukur] perkiraan 1 pelatihan ({n} esai x {resep['epochs']} epoch + prediksi): {menit:.0f} menit")
                    print("[ukur] prediksi diasumsikan 3x lebih cepat daripada latihan (perkiraan kasar).")
                    return None
        log["epoch_loss"].append(tot / max(cnt, 1))
        print(f"[{label} f{fold} s{seed}] epoch {ep + 1}/{epochs} loss {log['epoch_loss'][-1]:.4f} ({time.time() - t0:.0f} dtk)", flush=True)
    hasil = prediksi(model, flat, offs, pad, idx_uji, resep["eval_batch"], dev, amp)
    if jenis == "kelas":
        probs = hasil
        pred = probs @ C.SKALA
    else:
        probs = None
        pred = norm_ke_skor(hasil)
    log["pred_sd"] = float(np.std(pred))                             # ~0 berarti model runtuh (semua prediksi sama)
    log["detik"] = time.time() - t0
    log["puncak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
    if not opsi.smoke and log["pred_sd"] < 1e-3:
        raise RuntimeError(f"[{label} f{fold} s{seed}] GAGAL: prediksi hampir sama untuk semua esai (SD {log['pred_sd']:.2e}); model tidak belajar. Tidak disimpan.")
    return dict(pred=pred.astype(np.float64), probs=probs, log=log, idx_uji=idx_uji)
