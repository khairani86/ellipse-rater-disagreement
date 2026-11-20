"""K4 - skor terendah dari dua penilai (label satu kelas). Rujukan dan adaptasi sama dengan K3 (sisi sebaliknya)."""
import numpy as np

import common as C

KODE = "K4"
NAMA = C.NAMA["K4"]
RUJUKAN = ["Uto (2026), Lampiran B (aturan terkecil/terbesar pada skor gabungan ASAP asli)"]
PILIHAN_KITA = ["memakai aturan itu sebagai target latih pada ELLIPSE; skor acuan sendiri esai uji = skor terendah"]


def bangun(df, idx_latih, idx_uji, cfg):
    mn = np.minimum(df.x_a.values, df.x_b.values)
    return dict(idx=np.asarray(idx_latih), P=C.satu_kelas(mn[idx_latih]),
                acuan_uji=mn[idx_uji].astype(float), info=dict(n_latih=int(len(idx_latih))))
