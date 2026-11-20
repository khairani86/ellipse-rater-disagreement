"""jalankan_baseline.py - menjalankan SATU pembanding (B0 sampai B3), terpisah dari K2-K6. Semua pada target K2.

  B0 konstanta (CPU, detik)   B1 Ridge TF-IDF (CPU, beberapa menit per lipatan)
  B2 BERT standar (GPU)       B3 DeBERTa kepala regresi (GPU)

Contoh (PowerShell, satu perintah per baris):
  python jalankan_baseline.py --b B0
  python jalankan_baseline.py --b B1 --seed 0 --fold 0
  python umum/token_cache.py bert
  python jalankan_baseline.py --b B2 --seed 0 --fold 0
"""
import argparse
import os

import numpy as np

import baseline
import common as C
from umum import io_hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", required=True, choices=C.PEMBANDING + ["semua"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--ukur", action="store_true")
    ap.add_argument("--ukur-langkah", type=int, default=20, dest="ukur_langkah")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true", dest="no_grad_ckpt")
    ap.add_argument("--micro-batch", type=int, default=0, dest="micro_batch")
    ap.add_argument("--paksa", action="store_true")
    a = ap.parse_args()
    cfg = C.config()
    df = C.muat_baris()
    opsi = argparse.Namespace(smoke=a.smoke, cpu=a.cpu, no_grad_ckpt=a.no_grad_ckpt, micro_batch=a.micro_batch,
                              ukur=a.ukur, bench_steps=a.ukur_langkah)
    kodes = C.PEMBANDING if a.b == "semua" else [a.b]
    for kode in kodes:
        mod = baseline.muat(kode)
        for seed, fold in C.semua_kombinasi(cfg):
            if (a.seed is not None and seed != a.seed) or (a.fold is not None and fold != a.fold):
                continue
            fn = io_hasil.jalur_pred(kode, seed, fold, a.smoke)
            if os.path.exists(fn) and not (a.paksa or a.ukur):
                print(f"[{kode}] prediksi s{seed} f{fold} sudah ada, dilewati")
                continue
            idx_latih, idx_uji = C.bagi(len(df), seed, fold, cfg)
            out = mod.jalankan(df, idx_latih, idx_uji, seed, fold, cfg, opsi)
            if out is None:                                         # mode --ukur
                return
            acuan = df.rerata.values[out["idx_uji"]].astype(float)
            io_hasil.simpan_pred(kode, seed, fold, out, acuan, a.smoke)
            print(f"[{kode} f{fold} s{seed}] selesai; rerata prediksi {np.mean(out['pred']):.3f} SD {np.std(out['pred']):.3f}", flush=True)


if __name__ == "__main__":
    main()
