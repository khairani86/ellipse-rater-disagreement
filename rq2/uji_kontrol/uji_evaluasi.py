"""uji_evaluasi.py - memeriksa rumus ukuran kinerja dan inferensi terhadap jawaban yang diketahui (tanpa data ELLIPSE)."""
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

import _bantu as B
import common as C
from umum import evaluasi, uji_statistik
import pandas as pd

rs = np.random.default_rng(0)

# ---- pembulatan
B.periksa("kisi bulat: 3,5 -> 4 (0,5 naik), 3,49 -> 3, dipotong ke 1..5", list(C.half_up([3.5, 3.49, 0.2, 7.0, 1.5])) == [4, 3, 1, 5, 2])
B.periksa("kisi setengah: 3,26 -> 3,5; 3,24 -> 3,0; 3,75 -> 4,0; dipotong ke 1..5", list(C.kisi_setengah([3.26, 3.24, 3.75, 0.1, 9.0])) == [3.5, 3.0, 4.0, 1.0, 5.0])

# ---- QWK
a = rs.integers(1, 6, 2000)
b = np.clip(a + rs.integers(-1, 2, 2000), 1, 5)
ref = cohen_kappa_score(a, b, weights="quadratic")
B.periksa("QWK (kisi bulat) sama dengan scikit-learn pada label bulat", abs(C.qwk_skor(a.astype(float), b.astype(float), "bulat") - ref) < 1e-12, f"{ref:.4f}")
x = rs.integers(1, 6, 1000).astype(float)
B.periksa("QWK sempurna = 1 pada kedua kisi", abs(C.qwk_skor(x, x, "bulat") - 1) < 1e-12 and abs(C.qwk_skor(x, x, "setengah") - 1) < 1e-12)
B.periksa("QWK prediksi konstan = 0", abs(C.qwk_skor(np.full(1000, 3.0), x, "bulat")) < 1e-12 and abs(C.qwk_skor(np.full(1000, 3.0), x, "setengah")) < 1e-12)
m2 = (rs.integers(1, 6, 1500) + rs.integers(1, 6, 1500)) / 2
ia = np.rint(2 * (C.kisi_setengah(m2 + 0.1) - 1)).astype(int)
ib = np.rint(2 * (m2 - 1)).astype(int)
B.periksa("QWK kisi setengah = scikit-learn pada 9 tingkat", abs(C.qwk_skor(m2 + 0.1, m2, "setengah") - cohen_kappa_score(ia, ib, weights="quadratic")) < 1e-12)

# ---- PRMSE (Loukina dkk., 2020): simulasi dengan skor sejati yang diketahui
N = 200000
T = rs.normal(3, 0.6, N)
h1, h2 = T + rs.normal(0, 0.5, N), T + rs.normal(0, 0.5, N)
M = T + rs.normal(0, 0.3, N)
sejati = 1 - np.mean((T - M) ** 2) / np.var(T)
B.periksa("PRMSE dari dua penilai ~ PRMSE terhadap skor sejati yang diketahui", abs(C.prmse(M, h1, h2) - sejati) < 0.01, f"estimasi {C.prmse(M, h1, h2):.4f}, sejati {sejati:.4f}")
B.periksa("PRMSE prediksi sempurna (M = T) ~ 1", abs(C.prmse(T, h1, h2) - 1) < 0.01)
B.periksa("PRMSE konstanta = rerata ~ 0 atau negatif kecil", C.prmse(np.full(N, h1.mean()), h1, h2) < 0.01)

# ---- batas kappa (Uto 2026): simulasi dua penilai dengan reliabilitas yang diketahui
Tt = rs.normal(0, 1, 100000)
e1, e2 = Tt + rs.normal(0, 1, 100000), Tt + rs.normal(0, 1, 100000)
bk = C.batas_kappa(e1, e2)
B.periksa("rho_1 = ICC(1,1) = 0,5 untuk varians galat = varians sejati", abs(bk["rho1"] - 0.5) < 0.01, f"{bk['rho1']:.3f}")
B.periksa("kappa_HL <= kappa_max", bk["kappa_HL"] <= bk["kappa_max"] + 1e-12)
B.periksa("rho_Y = 2 rho_1 / (1 + rho_1)", abs(bk["rho_Y"] - 2 * bk["rho1"] / (1 + bk["rho1"])) < 1e-9)

# ---- Nadeau-Bengio dan Holm
d = rs.normal(0.02, 0.03, 15)
t_nb, p_nb, fak = C.nadeau_bengio(d, 1296, 5182)
t_biasa = stats.ttest_1samp(d, 0).statistic
B.periksa("t Nadeau-Bengio = t biasa / sqrt(1 + J n_uji / n_latih)", abs(t_nb - t_biasa / np.sqrt(1 + 15 * 1296 / 5182)) < 1e-9)
B.periksa("faktor SE = sqrt(1 + J n_uji / n_latih) = 2,18; setara 15 / 4,75 = 3,16 lipatan bebas", abs(fak - 2.18) < 0.005 and abs(15 / fak ** 2 - 3.16) < 0.01, f"{fak:.4f}, {15 / fak ** 2:.2f}")
B.periksa("p memakai derajat bebas J - 1", abs(p_nb - 2 * stats.t.sf(abs(t_nb), 14)) < 1e-12)
B.periksa("Holm: contoh [0,01; 0,04; 0,03; 0,005] -> [0,03; 0,06; 0,06; 0,02]", np.allclose(C.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02]))
B.periksa("Holm tidak pernah di atas 1 dan tidak di bawah p mentah", (lambda p: (C.holm(p) <= 1).all() and (C.holm(p) >= p).all())(rs.uniform(0, 1, 7)))

# ---- bandingkan(): tabel sintetis dengan selisih K3-K2 yang diketahui
baris = []
for s in range(3):
    for f in range(5):
        base = 0.70 + rs.normal(0, 0.01)
        for m, geser in (("K2", 0), ("K3", 0.05), ("K4", -0.05), ("K5", 0.0), ("K6", 0.0)):
            baris.append(dict(metode=m, seed=s, lipatan=f, n_latih=5182, n_uji=1296, qwk=base + geser + rs.normal(0, 0.003)))
h = uji_statistik.bandingkan(pd.DataFrame(baris), "qwk")
B.periksa("bandingkan: empat pembanding, 15 estimasi masing-masing", len(h) == 4 and (h.n_estimasi == 15).all())
B.periksa("bandingkan: K3 (+0,05) dan K4 (-0,05) nyata setelah Holm; K5 dan K6 (selisih 0) tidak",
          h.set_index("metode").p_holm[["K3", "K4"]].lt(0.001).all() and h.set_index("metode").p_holm[["K5", "K6"]].gt(0.05).all(), str(h.p_holm.round(4).tolist()))
B.periksa("bandingkan: selisih rerata K3 ~ +0,05", abs(h.set_index("metode").selisih_rerata["K3"] - 0.05) < 0.005)

# ---- metrik() end-to-end pada data kecil
df = pd.DataFrame(dict(x_a=rs.integers(1, 6, 400), x_b=rs.integers(1, 6, 400)))
mt = evaluasi.metrik(df, np.arange(400), (df.x_a + df.x_b) / 2, (df.x_a + df.x_b) / 2)
B.periksa("metrik: prediksi = rerata -> QWK rerata 1, korelasi 1, RMSE 0", abs(mt["qwk_rerata_setengah"] - 1) < 1e-12 and abs(mt["korelasi_rerata"] - 1) < 1e-12 and mt["rmse_rerata"] < 1e-12)
mk = evaluasi.metrik(df, np.arange(400), np.full(400, 3.0), np.full(400, 3.0))
B.periksa("metrik: prediksi konstan -> QWK 0, korelasi NaN (bukan galat)", abs(mk["qwk_rerata_bulat"]) < 1e-12 and np.isnan(mk["korelasi_rerata"]))
B.akhiri("uji_evaluasi")
