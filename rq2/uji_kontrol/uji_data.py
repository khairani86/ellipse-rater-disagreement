"""uji_data.py - himpunan resmi, lipatan, dan indeks penilai pada data ELLIPSE asli."""
import numpy as np

import _bantu as B
import common as C

cfg = C.config()
df = C.muat_baris()
B.periksa("6.478 esai resmi", len(df) == 6478, str(len(df)))
B.periksa("8.890 baris berkas, 6.482 ber-text_id_kaggle, 4 dibuang karena skor 0", (df.attrs["n_berkas"], df.attrs["n_ber_id"], df.attrs["n_dibuang_nol"]) == (8890, 6482, 4), str(df.attrs))
B.periksa("text_id_kaggle unik", df.text_id_kaggle.is_unique)
B.periksa("semua skor Overall 1..5 (tanpa 0)", df.x_a.between(1, 5).all() and df.x_b.between(1, 5).all())
B.periksa("18 esai selisih >= 2 dan tidak ada selisih >= 3", int((df.d >= 2).sum()) == 18 and int(df.d.max()) == 2, str(df.d.value_counts().to_dict()))
B.periksa("rerata = (x_a + x_b) / 2", np.allclose(df.rerata, (df.x_a + df.x_b) / 2))
raters, ia, ib, R = C.rater_arrays(df)
B.periksa("indeks penilai sah dan urutan penilai tetap (rater_1, rater_2, ...)", ia.max() < R and ib.max() < R and raters[0] == "rater_1", str(raters[:3]))

n = len(df)
sebelumnya = None
for seed in cfg["seeds"]:
    f = C.lipatan(n, seed, cfg)
    ukuran = np.bincount(f, minlength=cfg["n_folds"])
    B.periksa(f"seed {seed}: tiap esai tepat satu lipatan, ukuran lipatan seimbang", (f >= 0).all() and ukuran.max() - ukuran.min() <= 1, str(ukuran.tolist()))
    B.periksa(f"seed {seed}: deterministik (dua kali panggil sama)", np.array_equal(f, C.lipatan(n, seed, cfg)))
    tr, te = C.bagi(n, seed, 0, cfg)
    B.periksa(f"seed {seed}: latih dan uji terpisah dan lengkap", len(set(tr) & set(te)) == 0 and len(tr) + len(te) == n)
    if sebelumnya is not None:
        B.periksa(f"seed {seed}: pembagian lipatan BERBEDA dari seed sebelumnya", not np.array_equal(f, sebelumnya))
    sebelumnya = f
tr, te = C.bagi(n, 0, 0, cfg)
B.periksa("rasio uji/latih sekitar 0,25 (dasar faktor Nadeau-Bengio 2,18)", abs(len(te) / len(tr) - 0.25) < 0.001, f"{len(te) / len(tr):.4f}")
B.periksa("faktor SE Nadeau-Bengio untuk J = 15 dan rasio 0,25 = 2,18", abs(C.nadeau_bengio(np.arange(15.0), len(te), len(tr))[2] - 2.18) < 0.005, f"{C.nadeau_bengio(np.arange(15.0), len(te), len(tr))[2]:.4f}")
B.akhiri("uji_data")
