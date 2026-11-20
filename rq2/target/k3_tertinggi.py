"""K3 - skor tertinggi dari dua penilai (label satu kelas).

Rujukan  : Uto (2026), Lampiran B: skor gabungan ASAP Prompt 2-6 asli adalah skor terkecil atau terbesar dari dua penilai.
Adaptasi : Uto sendiri memakai rerata sebagai target; memakai tertinggi pada ELLIPSE adalah adaptasi kita.
"""
import numpy as np

import common as C

KODE = "K3"
NAMA = C.NAMA["K3"]
RUJUKAN = ["Uto (2026), Lampiran B (aturan terkecil/terbesar pada skor gabungan ASAP asli)"]
PILIHAN_KITA = ["memakai aturan itu sebagai target latih pada ELLIPSE; skor acuan sendiri esai uji = skor tertinggi"]


def bangun(df, idx_latih, idx_uji, cfg):
    mx = np.maximum(df.x_a.values, df.x_b.values)
    return dict(idx=np.asarray(idx_latih), P=C.satu_kelas(mx[idx_latih]),
                acuan_uji=mx[idx_uji].astype(float), info=dict(n_latih=int(len(idx_latih))))
