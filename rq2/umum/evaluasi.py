"""evaluasi.py - ukuran kinerja untuk SATU kombinasi (metode, seed, lipatan). Semua metode dan pembanding memakai fungsi yang sama.

Ukuran (semuanya pada esai uji lipatan itu; prediksi tidak dipotong lebih dulu):
  qwk_sendiri_{bulat,setengah} : QWK terhadap skor acuan SENDIRI metode itu (kisi pembulatan: lihat common.qwk_skor)
  qwk_rerata_{bulat,setengah}  : QWK terhadap rerata dua penilai (sasaran K2; hasil utama yang ditetapkan sebelumnya)
  korelasi_rerata, rmse_rerata : Pearson dan RMSE terhadap rerata dua penilai (prediksi tanpa dibulatkan)
  korelasi_sendiri, rmse_sendiri : idem terhadap skor acuan sendiri
  prmse                        : Loukina dkk. (2020), dua penilai per esai, M = prediksi tanpa dibulatkan
  kappa_max, kappa_HL          : batas Uto (2026) dari dua skor penilai pada esai uji itu
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402


def _korelasi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:               # prediksi konstan: korelasi tidak terdefinisi
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def metrik(df, idx_uji, pred, acuan_uji):
    idx_uji = np.asarray(idx_uji)
    pred = np.asarray(pred, float)
    acuan = np.asarray(acuan_uji, float)
    h1, h2 = df.x_a.values[idx_uji].astype(float), df.x_b.values[idx_uji].astype(float)
    rer = (h1 + h2) / 2
    out = {}
    for kisi in ("bulat", "setengah"):
        out[f"qwk_sendiri_{kisi}"] = C.qwk_skor(pred, acuan, kisi)
        out[f"qwk_rerata_{kisi}"] = C.qwk_skor(pred, rer, kisi)
    out["korelasi_rerata"] = _korelasi(pred, rer)
    out["rmse_rerata"] = _rmse(pred, rer)
    out["korelasi_sendiri"] = _korelasi(pred, acuan)
    out["rmse_sendiri"] = _rmse(pred, acuan)
    out["prmse"] = C.prmse(pred, h1, h2)
    b = C.batas_kappa(h1, h2)
    out["kappa_max"], out["kappa_HL"] = b["kappa_max"], b["kappa_HL"]
    return out
