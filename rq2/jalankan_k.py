"""jalankan_k.py - menjalankan SATU metode skor acuan (K2 sampai K6), terpisah dari yang lain.

Dua tahap yang bisa dijalankan sendiri-sendiri supaya terkontrol:
  --tahap target : membangun target latih dan skor acuan uji untuk tiap (seed, lipatan). Cepat, hanya CPU. Periksa dulu hasilnya.
  --tahap latih  : melatih DeBERTa pada target itu dan menyimpan prediksi esai uji. Perlu GPU (kecuali --smoke/--cpu).
  --tahap semua  : keduanya berurutan.

Contoh (PowerShell, satu perintah per baris):
  python jalankan_k.py --k K3 --tahap target
  python jalankan_k.py --k K3 --tahap latih --seed 0 --fold 0       # satu percobaan (uji coba per K)
  python jalankan_k.py --k K3 --tahap latih                         # 15 pelatihan (3 seed x 5 lipatan), yang sudah ada dilewati
  python jalankan_k.py --k K3 --tahap latih --ukur                  # ukur kecepatan dan memori, lalu keluar
"""
import argparse
import os
import sys

import numpy as np

import common as C
import target
from umum import io_hasil, latih, token_cache


def opsi_dari(a):
    return argparse.Namespace(smoke=a.smoke, cpu=a.cpu, no_grad_ckpt=a.no_grad_ckpt, micro_batch=a.micro_batch,
                              ukur=a.ukur, bench_steps=a.ukur_langkah)


def kombinasi(a, cfg):
    semua = C.semua_kombinasi(cfg)
    if a.seed is not None:
        semua = [(s, f) for s, f in semua if s == a.seed]
    if a.fold is not None:
        semua = [(s, f) for s, f in semua if f == a.fold]
    if not semua:
        raise SystemExit("kombinasi seed/lipatan tidak ada di config.json")
    return semua


def tahap_target(kode, df, cfg, a):
    mod = target.muat(kode)
    for seed, fold in kombinasi(a, cfg):
        fn = io_hasil.jalur_target(kode, seed, fold, a.smoke)
        if os.path.exists(fn) and not a.paksa:
            print(f"[{kode}] target s{seed} f{fold} sudah ada, dilewati")
            continue
        idx_latih, idx_uji = C.bagi(len(df), seed, fold, cfg)
        hasil = mod.bangun(df, idx_latih, idx_uji, cfg)
        assert np.allclose(hasil["P"].sum(1), 1.0) and np.isfinite(hasil["P"]).all(), "target tidak sah"
        io_hasil.simpan_target(kode, seed, fold, hasil, idx_uji, a.smoke)
        ev = hasil["P"] @ C.SKALA
        print(f"[{kode}] target s{seed} f{fold}: n_latih {len(hasil['idx'])}, nilai harapan rerata {ev.mean():.3f} SD {ev.std():.3f}; "
              f"acuan uji rerata {np.mean(hasil['acuan_uji']):.3f} SD {np.std(hasil['acuan_uji']):.3f}", flush=True)


def tahap_latih(kode, df, cfg, a):
    tok = token_cache.muat("deberta")
    resep = {k: cfg[k] for k in ("epochs", "micro_batch", "accum", "lr_backbone", "lr_head", "weight_decay", "warmup_ratio",
                                 "dropout", "grad_clip", "bf16", "grad_ckpt", "eval_batch")}
    for seed, fold in kombinasi(a, cfg):
        fn = io_hasil.jalur_pred(kode, seed, fold, a.smoke)
        if os.path.exists(fn) and not (a.paksa or a.ukur):
            print(f"[{kode}] prediksi s{seed} f{fold} sudah ada, dilewati")
            continue
        t = io_hasil.muat_target(kode, seed, fold, a.smoke)
        out = latih.latih_prediksi(tok, cfg["model_deberta"], "kelas", "rata", False, t["idx"], t["P"], t["idx_uji"],
                                   seed, fold, resep, opsi_dari(a), kode)
        if out is None:                                             # mode --ukur
            return
        acuan = t["acuan_uji"][:len(out["idx_uji"])]
        io_hasil.simpan_pred(kode, seed, fold, out, acuan, a.smoke)
        print(f"[{kode} f{fold} s{seed}] selesai {out['log']['detik'] / 60:.1f} menit (SD prediksi {out['log']['pred_sd']:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", required=True, choices=C.METODE + ["semua"])
    ap.add_argument("--tahap", default="semua", choices=["target", "latih", "semua"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--ukur", action="store_true", help="ukur kecepatan dan memori, lalu keluar")
    ap.add_argument("--ukur-langkah", type=int, default=20, dest="ukur_langkah")
    ap.add_argument("--smoke", action="store_true", help="uji ujung-ke-ujung dengan data sangat kecil (hasil ke folder smoke)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true", dest="no_grad_ckpt")
    ap.add_argument("--micro-batch", type=int, default=0, dest="micro_batch")
    ap.add_argument("--paksa", action="store_true", help="timpa hasil yang sudah ada")
    a = ap.parse_args()
    cfg = C.config()
    df = C.muat_baris()
    kodes = C.METODE if a.k == "semua" else [a.k]
    for kode in kodes:
        if a.tahap in ("target", "semua"):
            tahap_target(kode, df, cfg, a)
        if a.tahap in ("latih", "semua"):
            tahap_latih(kode, df, cfg, a)


if __name__ == "__main__":
    main()
