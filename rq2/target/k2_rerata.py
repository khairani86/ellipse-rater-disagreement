"""K2 - rerata dua penilai (pembanding bagi K3 sampai K6).

Rujukan  : Uto (2026): target = rerata dua penilai; Shabaruddin & Sim (2026): target = rerata dua penilai pada ELLIPSE.
Adaptasi : pemetaan ke kepala 5 logit (sebaran atas dua kelas bersebelahan dengan nilai harapan = rerata) pilihan kita.
"""
import numpy as np

import common as C

KODE = "K2"
NAMA = C.NAMA["K2"]
RUJUKAN = ["Uto (2026), arXiv 2604.19131: target = rerata dua penilai",
           "Shabaruddin & Sim (2026), Frontiers in Education 11, 1989989: target = rerata dua penilai (enam skor analitik)"]
PILIHAN_KITA = ["pemetaan rerata ke sebaran dua kelas bersebelahan (nilai harapan = rerata) agar kepala 5 logit sama untuk semua K"]


def bangun(df, idx_latih, idx_uji, cfg):
    m = df.rerata.values
    return dict(idx=np.asarray(idx_latih), P=C.dist_dari_rerata(m[idx_latih]),
                acuan_uji=m[idx_uji].astype(float), info=dict(n_latih=int(len(idx_latih))))
