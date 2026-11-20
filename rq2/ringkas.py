"""ringkas.py - mengumpulkan prediksi yang sudah ada, menghitung ukuran per (metode, seed, lipatan), lalu ringkasan dan uji terhadap K2.

Keluaran di hasil/:
  metrik_per_lipatan.csv   satu baris per (metode, seed, lipatan)
  ringkasan.csv / .md      rerata dan SD lintas seed dan lipatan (tiap estimasi = satu lipatan satu seed)
  uji_vs_K2.csv            selisih terhadap K2, t Nadeau-Bengio, p, p Holm (empat pembanding K3-K6), per ukuran
Hanya kombinasi yang sudah ada yang dihitung; kolom n_estimasi menunjukkan berapa dari 15 yang dipakai.
Pemakaian:  python ringkas.py          (python ringkas.py --smoke untuk hasil uji kecil)
"""
import argparse
import os

import numpy as np
import pandas as pd

import common as C
from umum import evaluasi, io_hasil, uji_statistik

UKURAN_UJI = ["qwk_rerata_bulat", "qwk_rerata_setengah", "qwk_sendiri_bulat", "qwk_sendiri_setengah", "prmse",
              "korelasi_rerata", "rmse_rerata"]
KOLOM_RINGKAS = ["qwk_rerata_bulat", "qwk_rerata_setengah", "qwk_sendiri_bulat", "qwk_sendiri_setengah", "prmse",
                 "korelasi_rerata", "rmse_rerata", "kappa_HL", "kappa_max"]


def kumpulkan(df, cfg, smoke):
    baris = []
    for kode in C.METODE + C.PEMBANDING:
        for seed, fold in C.semua_kombinasi(cfg):
            z = io_hasil.muat_pred(kode, seed, fold, smoke)
            if z is None:
                continue
            m = evaluasi.metrik(df, z["row_id"], z["pred"], z["acuan_uji"])
            n_uji = len(z["row_id"])
            baris.append(dict(metode=kode, seed=seed, lipatan=fold, n_uji=n_uji, n_latih=len(df) - n_uji, **m))
    return pd.DataFrame(baris)


def fmt_ms(x):
    x = np.asarray(x, float)
    if len(x) == 0 or np.all(np.isnan(x)):
        return "-"
    return f"{C.fmt(np.nanmean(x))} ± {C.fmt(np.nanstd(x, ddof=1) if len(x) > 1 else float('nan'))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = C.config()
    df = C.muat_baris()
    tabel = kumpulkan(df, cfg, a.smoke)
    if tabel.empty:
        raise SystemExit("Belum ada prediksi. Jalankan jalankan_k.py dan jalankan_baseline.py dulu.")
    out = C.wpath("hasil")
    os.makedirs(out, exist_ok=True)
    tabel.to_csv(os.path.join(out, "metrik_per_lipatan.csv"), index=False)

    urut = [k for k in C.METODE + C.PEMBANDING if k in set(tabel.metode)]
    ring = []
    for k in urut:
        t = tabel[tabel.metode == k]
        r = dict(metode=k, nama=C.NAMA[k], n_estimasi=len(t))
        for c in KOLOM_RINGKAS:
            r[c + "_rerata"] = t[c].mean()
            r[c + "_sd"] = t[c].std(ddof=1) if len(t) > 1 else np.nan
        ring.append(r)
    ring = pd.DataFrame(ring)
    ring.to_csv(os.path.join(out, "ringkasan.csv"), index=False)

    uji = []
    for u in UKURAN_UJI:
        h = uji_statistik.bandingkan(tabel, u)
        if len(h):
            h.insert(0, "ukuran", u)
            uji.append(h)
    uji = pd.concat(uji) if uji else pd.DataFrame()
    uji.to_csv(os.path.join(out, "uji_vs_K2.csv"), index=False)

    with open(os.path.join(out, "ringkasan.md"), "w", encoding="utf-8") as f:
        f.write("# Ringkasan RQ2 (rerata ± SD lintas seed dan lipatan)\n\n")
        f.write("Tiap estimasi = satu lipatan satu seed. QWK dihitung pada dua kisi pembulatan; kisi mana yang dipakai di makalah belum ditetapkan.\n\n")
        f.write("| Metode | n | " + " | ".join(KOLOM_RINGKAS) + " |\n|" + "---|" * (2 + len(KOLOM_RINGKAS)) + "\n")
        for k in urut:
            t = tabel[tabel.metode == k]
            f.write(f"| {k} | {len(t)} | " + " | ".join(fmt_ms(t[c]) for c in KOLOM_RINGKAS) + " |\n")
        f.write("\n# Uji terhadap K2 (Nadeau-Bengio, Holm pada K3-K6)\n\n")
        if len(uji):
            f.write("| Ukuran | Metode | selisih | t | p | p Holm | faktor SE |\n|---|---|---|---|---|---|---|\n")
            for _, r in uji.iterrows():
                f.write(f"| {r.ukuran} | {r.metode} | {C.fmt(r.selisih_rerata, 4)} | {C.fmt(r.t, 2)} | {C.fmt(r.p, 4)} | {C.fmt(r.p_holm, 4)} | {C.fmt(r.faktor_SE, 2)} |\n")
        else:
            f.write("(belum ada K2 dan lawannya)\n")
    print(open(os.path.join(out, "ringkasan.md"), encoding="utf-8").read())
    print("Tersimpan di", out)


if __name__ == "__main__":
    main()
