"""B0 - konstanta: prediksi = rerata target pada data latih lipatan itu.

Rujukan : Shabaruddin & Sim (2026), Bagian 4.6, B0: "Predicts each variable's training-fold mean; any model failing to beat it has learned nothing."
Pilihan kita: target = K2 (rerata dua penilai); tidak ada.
"""
import numpy as np

import common as C

KODE = "B0"
NAMA = C.NAMA["B0"]
RUJUKAN = ["Shabaruddin & Sim (2026), Frontiers in Education 11, 1989989, Bagian 4.6 (B0)"]
PILIHAN_KITA = ["target = K2 (rerata dua penilai), sama dengan K2-K6"]


def jalankan(df, idx_latih, idx_uji, seed, lipatan, cfg, opsi=None):
    mu = float(df.rerata.values[idx_latih].mean())
    return dict(pred=np.full(len(idx_uji), mu), probs=None, idx_uji=np.asarray(idx_uji), log=dict(label="B0", rerata_latih=mu))
