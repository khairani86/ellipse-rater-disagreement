"""K5 - seperti K2, tetapi esai yang selisih kedua skornya >= ambang tidak dipakai melatih (hanya pada data latih).

Rujukan  : Uma dkk. (2021), JAIR 72, Bagian 3.2 "Filtering Hard Items": melatih hanya pada subset berkesepakatan tinggi.
Adaptasi : Uma tidak menyebut ambang; ambang selisih >= 2 (config: ambang_selisih_K5) adalah pilihan kita.
           Esai uji TIDAK dibuang: semua metode dinilai pada esai uji yang sama. Skor acuan sendiri esai uji = rerata (sama dengan K2),
           karena K5 hanya mengubah isi data latih, bukan definisi skor.
"""
import numpy as np

import common as C

KODE = "K5"
NAMA = C.NAMA["K5"]
RUJUKAN = ["Uma dkk. (2021), JAIR 72, 1385-1470, Bagian 3.2 (menyaring butir sulit)"]
PILIHAN_KITA = ["ambang selisih >= 2", "esai uji tidak dibuang", "skor acuan sendiri esai uji = rerata (akibat mekanis; sama dengan K2)"]


def bangun(df, idx_latih, idx_uji, cfg):
    idx_latih = np.asarray(idx_latih)
    d = df.d.values
    simpan = d[idx_latih] < cfg["ambang_selisih_K5"]
    idx = idx_latih[simpan]
    m = df.rerata.values
    return dict(idx=idx, P=C.dist_dari_rerata(m[idx]), acuan_uji=m[idx_uji].astype(float),
                info=dict(n_latih=int(len(idx)), n_dibuang=int((~simpan).sum()), ambang=int(cfg["ambang_selisih_K5"])))
