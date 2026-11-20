"""uji_baseline.py - uji kontrol B0 dan B1 pada data ELLIPSE asli (CPU). B2 dan B3 diuji di uji_ujung_ke_ujung.py (perlu model mini).

Pemeriksaan: nilai yang diharapkan, bentuk, deterministik, dan TIDAK BOCOR: skor esai uji diubah -> prediksi tidak berubah;
teks esai uji lain diubah -> prediksi esai uji lainnya tidak berubah (TF-IDF dan IDF hanya ditaksir dari data latih).
"""
import time

import numpy as np

import _bantu as B
import baseline
import common as C

cfg = C.config()
df0 = C.muat_baris()
n = len(df0)
tr, te = C.bagi(n, 0, 0, cfg)

# ---------------------------------------------------------------- B0
b0 = baseline.muat("B0")
o = b0.jalankan(df0, tr, te, 0, 0, cfg)
B.periksa("B0: prediksi konstan sama dengan rerata target K2 pada data latih", np.allclose(o["pred"], df0.rerata.values[tr].mean()) and len(o["pred"]) == len(te))
d2 = df0.copy()
d2.loc[te, ["x_a", "x_b"]] = 1
d2["rerata"] = (d2.x_a + d2.x_b) / 2
B.periksa("B0: TIDAK BOCOR (skor esai uji diubah -> prediksi sama)", np.allclose(b0.jalankan(d2, tr, te, 0, 0, cfg)["pred"], o["pred"]))

# ---------------------------------------------------------------- B1 (subset kecil agar cepat)
b1 = baseline.muat("B1")
rs = np.random.default_rng(5)
tr_k, te_k = tr[:900], te[:250]
t0 = time.time()
o1 = b1.jalankan(df0, tr_k, te_k, 0, 0, cfg)
print(f"[CATAT] B1 pada 900 latih / 250 uji: {time.time() - t0:.1f} detik; alpha terpilih {o1['log']['alpha']}, fitur {o1['log']['n_fitur']}")
B.periksa("B1: alpha terpilih berasal dari kisi", o1["log"]["alpha"] in cfg["ridge_alpha"])
B.periksa("B1: prediksi berhingga, berbentuk (n_uji,), tidak dipotong (bisa di luar 1..5 bila model begitu)", o1["pred"].shape == (len(te_k),) and np.isfinite(o1["pred"]).all())
B.periksa("B1: prediksi berkorelasi positif dengan rerata dua penilai pada esai uji", np.corrcoef(o1["pred"], df0.rerata.values[te_k])[0, 1] > 0.3, f"r = {np.corrcoef(o1['pred'], df0.rerata.values[te_k])[0, 1]:.2f}")
o1b = b1.jalankan(df0, tr_k, te_k, 0, 0, cfg)
B.periksa("B1: deterministik", np.allclose(o1["pred"], o1b["pred"]) and o1["log"]["alpha"] == o1b["log"]["alpha"])
d3 = df0.copy()
d3.loc[te_k, ["x_a", "x_b"]] = rs.integers(1, 6, (len(te_k), 2))
d3["rerata"] = (d3.x_a + d3.x_b) / 2
B.periksa("B1: TIDAK BOCOR (skor esai uji diacak -> prediksi sama)", np.allclose(b1.jalankan(d3, tr_k, te_k, 0, 0, cfg)["pred"], o1["pred"]))
d4 = df0.copy()
d4.loc[te_k[0], "Text"] = "kata baru yang tidak pernah ada di data latih " * 20
o4 = b1.jalankan(d4, tr_k, te_k, 0, 0, cfg)
B.periksa("B1: mengubah teks SATU esai uji tidak mengubah prediksi esai uji lain (IDF hanya dari data latih)", np.allclose(o4["pred"][1:], o1["pred"][1:]) and not np.isclose(o4["pred"][0], o1["pred"][0]))
d5 = df0.copy()
d5.loc[tr_k[0], "rerata"] = 5.0
B.periksa("B1: mengubah target latih mengubah prediksi (target latih memang dipakai)", not np.allclose(b1.jalankan(d5, tr_k, te_k, 0, 0, cfg)["pred"], o1["pred"]))
alpha_dalam = b1.pilih_alpha([str(t) for t in df0.Text.values[tr_k]], df0.rerata.values[tr_k].astype(float), cfg, 0)[0]
B.periksa("B1: pemilihan alpha hanya memakai data latih (hasil sama dengan yang dicatat)", alpha_dalam == o1["log"]["alpha"])
B.akhiri("uji_baseline")
