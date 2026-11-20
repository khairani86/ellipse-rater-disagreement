"""uji_statistik.py - perbandingan berpasangan terhadap K2 dengan uji-t sampel berulang terkoreksi Nadeau-Bengio
dan koreksi Holm-Bonferroni dalam SATU keluarga (K3 sampai K6 terhadap K2 = empat pembanding).

Rujukan: Shabaruddin & Sim (2026): uji-t terkoreksi Nadeau-Bengio atas 15 estimasi, Holm-Bonferroni pada keluarga yang ditetapkan sebelumnya.
Rumus: common.nadeau_bengio dan common.holm, diuji pada uji_kontrol/uji_evaluasi.py (makalah asli Nadeau-Bengio dan Holm belum kita buka).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402


def bandingkan(tabel, ukuran, acuan="K2", lawan=("K3", "K4", "K5", "K6")):
    """tabel: DataFrame dengan kolom metode, seed, lipatan, n_latih, n_uji, dan kolom `ukuran`.
    Mengembalikan DataFrame: selisih rerata (lawan - acuan), t, p, p Holm, faktor SE."""
    wide = tabel.pivot_table(index=["seed", "lipatan"], columns="metode", values=ukuran)
    ukr = tabel[tabel.metode == acuan].set_index(["seed", "lipatan"])[["n_latih", "n_uji"]]
    baris = []
    for m in lawan:
        if m not in wide.columns or acuan not in wide.columns:
            continue
        d = (wide[m] - wide[acuan]).dropna()
        u = ukr.loc[d.index]
        t, p, fak = C.nadeau_bengio(d.values, float(u.n_uji.mean()), float(u.n_latih.mean()))
        baris.append(dict(metode=m, n_estimasi=len(d), selisih_rerata=float(d.mean()), selisih_sd=float(d.std(ddof=1)), t=t, p=p, faktor_SE=fak))
    out = pd.DataFrame(baris)
    if len(out):
        out["p_holm"] = C.holm(out.p.fillna(1.0).values)
    return out
