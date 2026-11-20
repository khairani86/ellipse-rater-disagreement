"""uji_target.py - uji kontrol PER K pada data ELLIPSE asli (CPU, detik). Tiap K diuji sendiri.

Pemeriksaan umum (semua K): bentuk, jumlah baris = 1, tidak negatif, tidak ada NaN, deterministik, idx subset esai latih,
dan TIDAK BOCOR: bila skor esai uji diacak, target latih tidak boleh berubah sedikit pun.
Pemeriksaan khusus tiap K dijelaskan di fungsinya.
Pemakaian:  python uji_kontrol/uji_target.py            (semua K)
            python uji_kontrol/uji_target.py K6         (satu K)
"""
import sys

import numpy as np
from scipy import stats

import _bantu as B
import common as C
import target

cfg = C.config()
df0 = C.muat_baris()
n = len(df0)
pilih = sys.argv[1:] or C.METODE
KOMBINASI = [(0, 0), (1, 2)]                       # dua (seed, lipatan) berbeda; uji khusus K6 memakai tambahan


def acak_skor_uji(df, idx_uji, rs):
    """Salinan df dengan skor kedua penilai pada esai uji diganti skor acak 1..5."""
    d = df.copy()
    d.loc[idx_uji, "x_a"] = rs.integers(1, 6, len(idx_uji))
    d.loc[idx_uji, "x_b"] = rs.integers(1, 6, len(idx_uji))
    d["d"] = (d.x_a - d.x_b).abs()
    d["rerata"] = (d.x_a + d.x_b) / 2
    return d


def umum(kode):
    mod = target.muat(kode)
    for seed, fold in KOMBINASI:
        tr, te = C.bagi(n, seed, fold, cfg)
        r = mod.bangun(df0, tr, te, cfg)
        P = r["P"]
        tag = f"{kode} s{seed} f{fold}"
        B.periksa(f"{tag}: P berbentuk (n_dipakai, 5), tiap baris berjumlah 1, tidak negatif, tanpa NaN",
                  P.shape == (len(r["idx"]), 5) and np.allclose(P.sum(1), 1) and (P >= -1e-12).all() and np.isfinite(P).all())
        B.periksa(f"{tag}: idx subset esai latih dan tidak memuat esai uji", set(r["idx"]) <= set(tr) and not (set(r["idx"]) & set(te)))
        B.periksa(f"{tag}: acuan_uji ada untuk tiap esai uji, berhingga, dalam [1, 5]", len(r["acuan_uji"]) == len(te) and np.isfinite(r["acuan_uji"]).all()
                  and r["acuan_uji"].min() >= 1 - 1e-9 and r["acuan_uji"].max() <= 5 + 1e-9)
        r2 = mod.bangun(df0, tr, te, cfg)
        B.periksa(f"{tag}: deterministik", np.array_equal(P, r2["P"]) and np.array_equal(r["acuan_uji"], r2["acuan_uji"]))
        rs = np.random.default_rng(123)
        r3 = mod.bangun(acak_skor_uji(df0, te, rs), tr, te, cfg)
        B.periksa(f"{tag}: TIDAK BOCOR (skor esai uji diacak -> target latih identik)", np.array_equal(P, r3["P"]) and np.array_equal(r["idx"], r3["idx"]))
    return mod


def uji_k2():
    tr, te = C.bagi(n, 0, 0, cfg)
    r = target.muat("K2").bangun(df0, tr, te, cfg)
    ev = r["P"] @ C.SKALA
    B.periksa("K2: nilai harapan target = rerata dua skor (tepat)", np.allclose(ev, df0.rerata.values[tr]))
    B.periksa("K2: paling banyak dua kelas bersebelahan berisi massa", ((r["P"] > 0).sum(1) <= 2).all())
    sel = np.where((r["P"] > 0).sum(1) == 2)[0]
    kelas = np.array([np.where(r["P"][i] > 0)[0] for i in sel])
    B.periksa("K2: dua kelas itu bersebelahan, massa 0,5/0,5 untuk rerata .5", (kelas[:, 1] - kelas[:, 0] == 1).all() and np.allclose(r["P"][sel][r["P"][sel] > 0], 0.5))
    B.periksa("K2: acuan_uji = rerata dua skor", np.allclose(r["acuan_uji"], df0.rerata.values[te]))


def uji_k3_k4():
    tr, te = C.bagi(n, 0, 0, cfg)
    r3 = target.muat("K3").bangun(df0, tr, te, cfg)
    r4 = target.muat("K4").bangun(df0, tr, te, cfg)
    r2 = target.muat("K2").bangun(df0, tr, te, cfg)
    e3, e4, e2 = r3["P"] @ C.SKALA, r4["P"] @ C.SKALA, r2["P"] @ C.SKALA
    B.periksa("K3 dan K4: label satu kelas (one-hot)", ((r3["P"] == 1).sum(1) == 1).all() and ((r4["P"] == 1).sum(1) == 1).all())
    B.periksa("K3 = skor tertinggi, K4 = skor terendah dari dua penilai",
              np.array_equal(e3, np.maximum(df0.x_a.values[tr], df0.x_b.values[tr])) and np.array_equal(e4, np.minimum(df0.x_a.values[tr], df0.x_b.values[tr])))
    B.periksa("K4 <= K2 <= K3 pada tiap esai", (e4 <= e2 + 1e-12).all() and (e2 <= e3 + 1e-12).all())
    d = df0.d.values[tr]
    B.periksa("K3 - K4 = selisih dua penilai", np.array_equal(e3 - e4, d))
    B.periksa("K3 = K4 = K2 bila kedua penilai sama", np.allclose(e3[d == 0], e2[d == 0]) and np.allclose(e4[d == 0], e2[d == 0]))
    B.periksa("K3 dan K4: acuan_uji = tertinggi / terendah", np.array_equal(r3["acuan_uji"], np.maximum(df0.x_a.values[te], df0.x_b.values[te]))
              and np.array_equal(r4["acuan_uji"], np.minimum(df0.x_a.values[te], df0.x_b.values[te])))


def uji_k5():
    total = 0
    for seed, fold in [(0, 0), (0, 1), (2, 4)]:
        tr, te = C.bagi(n, seed, fold, cfg)
        r5 = target.muat("K5").bangun(df0, tr, te, cfg)
        r2 = target.muat("K2").bangun(df0, tr, te, cfg)
        dibuang = np.setdiff1d(tr, r5["idx"])
        B.periksa(f"K5 s{seed} f{fold}: yang dibuang tepat esai latih berselisih >= 2", np.array_equal(np.sort(dibuang), np.sort(tr[df0.d.values[tr] >= 2])),
                  f"{len(dibuang)} dibuang")
        B.periksa(f"K5 s{seed} f{fold}: target sisa sama dengan K2 pada esai itu", np.allclose(r5["P"], r2["P"][np.isin(tr, r5["idx"])]))
        B.periksa(f"K5 s{seed} f{fold}: esai uji TIDAK dibuang, acuan_uji = rerata (sama dengan K2)", np.allclose(r5["acuan_uji"], r2["acuan_uji"]) and len(r5["acuan_uji"]) == len(te))
        total += len(dibuang)
    tr0, te0 = C.bagi(n, 0, 0, cfg)
    B.periksa("K5: di tiap lipatan yang dibuang <= 18 (18 esai berselisih >= 2 di seluruh data)", total <= 3 * 18)
    # ambang diubah -> yang dibuang berubah (ambang benar-benar dipakai)
    cfg3 = dict(cfg, ambang_selisih_K5=3)
    r = target.muat("K5").bangun(df0, tr0, te0, cfg3)
    B.periksa("K5: ambang 3 tidak membuang apa pun (tidak ada selisih >= 3)", len(r["idx"]) == len(tr0))


def uji_k6():
    import mfrm_fast as F
    from target import k6_rasch_keketatan as K6
    tr, te = C.bagi(n, 0, 0, cfg)
    r = K6.bangun(df0, tr, te, cfg)
    par, P = r["_par"], r["P"]
    info = r["info"]
    B.periksa("K6: pengoptimal konvergen dan gradien kecil", info["sukses"] and info["grad_maks"] < 0.05, f"grad maks {info['grad_maks']:.1e}, {info['iterasi']} iterasi")
    B.periksa("K6: semua 27 penilai punya data di lipatan latih dan dipakai", info["n_penilai_dipakai"] == par["R"] == 27, str(info["n_penilai_dipakai"]))
    ev = P @ C.SKALA
    B.periksa("K6: nilai harapan target dalam (1, 5) dan urutannya mengikuti rerata skor (Spearman >= 0,9)",
              ev.min() > 1 and ev.max() < 5 and stats.spearmanr(ev, df0.rerata.values[tr])[0] >= 0.9, f"rho {stats.spearmanr(ev, df0.rerata.values[tr])[0]:.3f}")
    # esai ekstrem: JMLE tidak punya theta berhingga untuk 5-5 dan 1-1, EAP harus punya
    ekstrem = np.where(((df0.x_a.values[tr] == 5) & (df0.x_b.values[tr] == 5)) | ((df0.x_a.values[tr] == 1) & (df0.x_b.values[tr] == 1)))[0]
    B.periksa("K6: esai berskor 5-5 dan 1-1 punya theta berhingga dan target sah", len(ekstrem) > 0 and np.isfinite(r["_theta_latih"][ekstrem]).all() and np.isfinite(P[ekstrem]).all(), f"{len(ekstrem)} esai")
    # arah keketatan: penilai yang cenderung memberi skor lebih rendah dari mitranya harus lebih ketat (s lebih besar)
    raters, ia, ib, R = C.rater_arrays(df0)
    sel = np.zeros(R)
    cnt = np.zeros(R)
    for i_, x, y in ((ia, df0.x_a.values, df0.x_b.values), (ib, df0.x_b.values, df0.x_a.values)):
        sel += np.bincount(i_[tr], weights=(x - y)[tr], minlength=R)
        cnt += np.bincount(i_[tr], minlength=R)
    rho = stats.spearmanr(par["s"], sel / cnt)[0]
    B.periksa("K6: keketatan berkorelasi negatif dengan 'skor yang diberi dikurangi skor mitra'", rho < -0.3, f"rho {rho:.2f}")
    # simetri slot: menukar penilai pertama dan kedua tidak mengubah apa pun
    d2 = df0.copy()
    d2["Rater_1"], d2["Rater_2"] = df0.Rater_2.values, df0.Rater_1.values
    d2["x_a"], d2["x_b"] = df0.x_b.values, df0.x_a.values
    r_sw = K6.bangun(d2, tr, te, cfg)
    B.periksa("K6: simetris terhadap urutan slot penilai (tukar slot -> target sama)", np.allclose(r_sw["P"], P, atol=1e-4) and np.allclose(r_sw["acuan_uji"], r["acuan_uji"], atol=1e-4),
              f"selisih maks {np.abs(r_sw['P'] - P).max():.1e}")
    # tidak bocor: skor uji diacak -> parameter model dan target latih tidak berubah; acuan uji berubah
    rs = np.random.default_rng(7)
    r_x = K6.bangun(acak_skor_uji(df0, te, rs), tr, te, cfg)
    B.periksa("K6: parameter model (keketatan, ambang) tidak berubah bila skor uji diacak", np.allclose(r_x["_par"]["s"], par["s"]) and np.allclose(r_x["_par"]["tau"], par["tau"]))
    B.periksa("K6: acuan_uji ikut berubah bila skor uji berubah (diduga dari skor esai uji itu sendiri)", not np.allclose(r_x["acuan_uji"], r["acuan_uji"]))
    # EAP: cocokkan dengan integrasi numerik terpisah untuk beberapa esai
    xa, xb = df0.x_a.values - 1, df0.x_b.values - 1
    cum = np.concatenate([[0.0], np.cumsum(par["tau"])])
    grid = np.linspace(par["mu"] - 8 * par["sig"], par["mu"] + 8 * par["sig"], 4001)
    ok = True
    for i in tr[:6]:
        lik = np.exp(stats.norm.logpdf(grid, par["mu"], par["sig"]))
        for rr, x in ((par["ia"][i], xa[i]), (par["ib"][i], xb[i])):
            u = np.arange(5)[None, :] * (grid[:, None] - par["s"][rr]) - cum[None, :]
            lik = lik * np.exp(u[:, x] - np.logaddexp.reduce(u, axis=1))
        th = (grid * lik).sum() / lik.sum()
        ok &= abs(th - r["_theta_latih"][np.where(tr == i)[0][0]]) < 0.02
    B.periksa("K6: theta EAP cocok dengan integrasi numerik terpisah", ok)
    # sebaran target = rerata atas penilai dari rumus langsung (satu esai, tanpa vektorisasi)
    i = 3
    th = r["_theta_latih"][i]
    Ps = []
    for rr in np.where(par["n_r"] > 0)[0]:
        u = np.arange(5) * (th - par["s"][rr]) - cum
        Ps.append(np.exp(u - np.logaddexp.reduce(u)))
    B.periksa("K6: target satu esai = rerata atas penilai dari rumus langsung", np.allclose(np.mean(Ps, axis=0), P[i], atol=1e-9))
    # dua esai dengan penilai dan skor yang sama mendapat target yang sama
    kunci = list(zip(par["ia"][tr], par["ib"][tr], xa[tr], xb[tr]))
    dup = {}
    for j, k in enumerate(kunci):
        dup.setdefault(k, []).append(j)
    pasangan = next(v for v in dup.values() if len(v) >= 2)
    B.periksa("K6: esai dengan penilai dan skor sama -> target sama", np.allclose(P[pasangan[0]], P[pasangan[1]]))
    # kompresi skala: SD nilai harapan K6 lebih kecil dari K2 (penyusutan EAP + rerata penilai); dicatat, bukan kegagalan
    print(f"[CATAT] SD nilai harapan target: K6 {ev.std():.3f}, K2 {df0.rerata.values[tr].std():.3f}")
    print(f"[CATAT] keketatan penilai: min {info['keketatan_min']:.2f}, maks {info['keketatan_maks']:.2f}; sigma theta {info['sigma']:.2f}")
    # beberapa (seed, lipatan) lain: tetap konvergen
    sukses = []
    for seed, fold in [(0, 1), (1, 3), (2, 4)]:
        tr2, te2 = C.bagi(n, seed, fold, cfg)
        i2 = K6.bangun(df0, tr2, te2, cfg)["info"]
        sukses.append(i2["sukses"] and i2["grad_maks"] < 0.05)
    B.periksa("K6: konvergen pada tiga (seed, lipatan) lain", all(sukses))


for kode in pilih:
    umum(kode)
    {"K2": uji_k2, "K3": uji_k3_k4, "K4": lambda: None, "K5": uji_k5, "K6": uji_k6}[kode]()
B.akhiri("uji_target")
