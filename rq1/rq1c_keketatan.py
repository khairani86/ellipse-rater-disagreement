"""rq1c_keketatan.py - RQ1c: apakah beda keketatan antar penilai lebih besar daripada beda kesulitan antar aspek?

Satu berkas, hanya butuh numpy dan pandas. Tidak memakai berkas lain di paket.

Model: many-facet Rasch kredit parsial, estimasi JMLE (cara FACETS). Kim (2026) memakai JMLE di FACETS 3.87.0 dan tidak menyebut
jenis model utamanya; kredit parsial adalah pilihan kami (Kim memakainya untuk pemeriksaan silang kesulitan kriteria).

    log P(skor = k) / P(skor = k-1) = theta_i - s_r - d_j - tau_jk ,   k = 2, 3, 4, 5  (4 ambang per kriteria)

    theta_i  kemampuan esai i        s_r  keketatan penilai r (positif = lebih ketat)
    d_j      kesulitan kriteria j (positif = lebih sulit)       tau_jk  ambang langkah kriteria j; jumlah tau_jk = 0
Identifikasi: rerata s_r = 0 dan rerata d_j = 0. Esai yang semua skornya terendah atau semua tertinggi dikeluarkan (seperti FACETS).

Dua versi dijalankan berdampingan:
    a6   enam aspek analitik (Cohesion, Syntax, Vocabulary, Phraseology, Grammar, Conventions)
    a6o  enam aspek + Overall sebagai kriteria ke-7 (Kim memasukkan kriteria holistik; hasilnya sangat bergantung padanya)

Keluaran: keketatan penilai dan kesulitan aspek (logit), rentang, simpangan baku, rasio, pemisahan dan keandalan, kecocokan
(infit/outfit), geser skor dalam poin, pemeriksaan silang dengan penaksir linear (Kim), bias penilai x aspek (Wu dkk., 2026),
dan sensitivitas pada tiga himpunan baris.

Himpunan utama: baris ber-text_id_kaggle (himpunan resmi). Sensitivitas: semua baris yang dipakai (Overall > 0 pada kedua
penilai) dan baris tanpa text_id_kaggle. Himpunan utama bisa diganti dengan --utama semua.

Pakai:  python rq1c_keketatan.py [csv] [folder_keluaran] [--rep N] [--utama resmi|semua]
        (bawaan: data/ellipse_raw.csv, hasil_rq1c/, N = 3 ulangan uji pemulihan parameter, himpunan utama resmi)
"""
import math
import os
import sys

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):                 # supaya konsol Windows tidak berhenti karena karakter khusus
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261115
B_MC = 2000                                         # undian Monte Carlo dari galat baku (ketidakpastian rentang dan rasio)
KC = 5                                              # kategori 0..4 = skor 1..5
KS = np.arange(KC)
ASPEK = ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]
VERSI = {"a6": ASPEK, "a6o": ASPEK + ["Overall"]}
NAMA_VERSI = {"a6": "6 aspek", "a6o": "6 aspek + Overall"}
UTAMA = "resmi"                                     # himpunan utama: "resmi" (baris ber-text_id_kaggle) atau "semua"
HIMPUNAN = {"resmi": "Baris ber-text_id_kaggle (himpunan resmi)", "semua": "Semua baris", "tanpa": "Baris tanpa text_id_kaggle"}
MIN_ESAI = 100                                      # penilai dengan esai >= ini dipakai untuk rentang pembanding
BIAS_MIN_OBS = 30                                   # sel penilai x aspek dengan pengamatan >= ini dianalisis
BIAS_T, BIAS_ABS = 2.0, 0.5                         # tanda bias menurut Wu dkk. (2026): |t| > 2 dan |bias| > 0,5 logit
FIT_KIM, FIT_WU = (0.5, 1.5), (0.6, 1.4)            # rentang infit/outfit yang dapat diterima: Kim (2026); Wu dkk. (2026)
# Angka dari Kim (2026), untuk konteks (bukan hasil kita): Bagian 4.5, Tabel 8 dan 9
KIM = dict(rentang_penilai=1.003, G_penilai=7.33, R_penilai=0.982, rentang_kriteria=3.09,
           holistik=2.108, analitik_maks=0.569, analitik_min=-0.981)


# ----------------------------------------------------------------------------------------------------------
# 1. Data dan pengamatan
# ----------------------------------------------------------------------------------------------------------
def no(s):                                          # "rater_9" -> 9, supaya urutan penilai selalu sama
    try:
        return int(str(s).split("_")[-1])
    except ValueError:
        return 10 ** 9


def muat(csv):
    """Baca berkas mentah. Baris dipakai bila skor Overall kedua penilai > 0 (0 = tidak ada skor); sama dengan RQ1a dan RQ1b."""
    if not os.path.exists(csv):
        raise SystemExit(f"Berkas tidak ditemukan: {csv}\nTaruh berkas ELLIPSE mentah di data/ellipse_raw.csv")
    df = pd.read_csv(csv)
    perlu = ["Filename", "Rater_1", "Rater_2"] + [f"{c}_{s}" for c in VERSI["a6o"] for s in (1, 2)]
    hilang = [c for c in perlu if c not in df.columns]
    if hilang:
        raise SystemExit(f"Kolom tidak ada di berkas: {hilang}")
    n_awal = len(df)
    df = df[(df.Overall_1 > 0) & (df.Overall_2 > 0)].reset_index(drop=True)
    print(f"Baris di berkas: {n_awal}; dipakai (Overall > 0 pada kedua penilai): {len(df)}; esai unik (Filename): {df.Filename.nunique()}")
    return df


def mask_himpunan(df, kunci):
    ada = df.text_id_kaggle.notna().values
    return {"resmi": ada, "semua": np.ones(len(df), bool), "tanpa": ~ada}[kunci]


def pilih_utama(df):
    """Ambil himpunan utama dari semua baris yang dipakai (Overall > 0 pada kedua penilai)."""
    if UTAMA == "semua":
        return df
    if "text_id_kaggle" not in df.columns:
        raise SystemExit("Kolom text_id_kaggle tidak ada, jadi himpunan resmi tidak bisa dibentuk. Jalankan dengan --utama semua.")
    d = df[mask_himpunan(df, "resmi")].reset_index(drop=True)
    print(f"Himpunan utama ({HIMPUNAN['resmi']}): {len(d)} dari {len(df)} baris; esai unik (Filename): {d.Filename.nunique()}")
    return d


def bangun_obs(df, kriteria):
    """Satu pengamatan = satu skor > 0 untuk (esai, penilai, kriteria). Satu esai = satu Filename."""
    penilai = sorted(set(df.Rater_1) | set(df.Rater_2), key=no)
    pi = {r: i for i, r in enumerate(penilai)}
    eid, _ = pd.factorize(df.Filename.astype(str))
    E, R, J, X = [], [], [], []
    for jj, c in enumerate(kriteria):
        for slot in (1, 2):
            v = df[f"{c}_{slot}"].values.astype(int)
            m = v > 0
            E.append(eid[m])
            R.append(df[f"Rater_{slot}"].map(pi).values[m])
            J.append(np.full(int(m.sum()), jj))
            X.append(v[m] - 1)
    return np.concatenate(E), np.concatenate(R), np.concatenate(J), np.concatenate(X), penilai


def komponen(r, e, n_r):
    """Label komponen terhubung tiap penilai: dua penilai terhubung bila menilai esai yang sama, langsung atau lewat rantai."""
    induk = list(range(n_r + int(e.max()) + 1))

    def cari(a):
        while induk[a] != a:
            induk[a] = induk[induk[a]]
            a = induk[a]
        return a

    for rr, ee in np.unique(np.column_stack([r, e]), axis=0):
        induk[cari(int(rr))] = cari(n_r + int(ee))
    return np.array([cari(i) for i in range(n_r)])


def siapkan(df, kriteria):
    """Pengamatan siap estimasi: esai skor ekstrem dikeluarkan, hanya komponen terhubung terbesar dipakai, indeks dipadatkan."""
    e, r, j, x, penilai = bangun_obs(df, kriteria)
    n_awal = len(x)
    n_e = int(e.max()) + 1
    cnt = np.bincount(e, minlength=n_e)
    ekstrem = (cnt > 0) & ((np.bincount(e, x == 0, n_e) == cnt) | (np.bincount(e, x == 4, n_e) == cnt))
    k = ~ekstrem[e]
    n_ekstrem_obs = int((~k).sum())
    e, r, j, x = e[k], r[k], j[k], x[k]
    lab = komponen(r, e, len(penilai))
    n_komp = len(np.unique(lab[np.unique(r)]))
    terbesar = pd.Series(lab[r]).value_counts().index[0]
    k = lab[r] == terbesar
    n_dibuang_komp = int((~k).sum())
    e, r, j, x = e[k], r[k], j[k], x[k]
    _, e = np.unique(e, return_inverse=True)
    ur, r = np.unique(r, return_inverse=True)
    return dict(e=e, r=r, j=j, x=x, penilai=[penilai[i] for i in ur], n_e=int(e.max()) + 1, n_r=len(ur), n_j=len(kriteria),
                n_awal=n_awal, n_ekstrem=int(ekstrem.sum()), n_ekstrem_obs=n_ekstrem_obs, n_komp=int(n_komp),
                n_dibuang_komp=n_dibuang_komp)


# ----------------------------------------------------------------------------------------------------------
# 2. Model many-facet Rasch kredit parsial (JMLE)
# ----------------------------------------------------------------------------------------------------------
class MFRM:
    def __init__(self, e, r, j, x, n_e, n_r, n_j):
        self.e, self.r, self.j, self.x = e, r, j, x
        self.nE, self.nR, self.nJ = n_e, n_r, n_j
        self.s = np.zeros(n_r)
        self.d = np.zeros(n_j)
        self.tau = np.zeros((n_j, KC - 1))
        for jj in range(n_j):                       # nilai awal ambang dari proporsi kategori tiap kriteria
            p = np.bincount(x[j == jj], minlength=KC) / max((j == jj).sum(), 1) + 1e-3
            t = np.log(p[:-1] / p[1:])
            self.tau[jj] = t - t.mean()
        xb = np.bincount(e, weights=x, minlength=n_e) / np.maximum(np.bincount(e, minlength=n_e), 1)
        self.th = np.clip(1.2 * (xb - x.mean()), -3, 3)

    def cum(self, tau=None):
        tau = self.tau if tau is None else tau
        return np.concatenate([np.zeros((self.nJ, 1)), np.cumsum(tau, axis=1)], axis=1)

    def probs(self, eta, j, tau=None):
        z = eta[:, None] * KS[None, :] - self.cum(tau)[j]
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def moments(self, th=None, s=None, d=None, tau=None):
        th = self.th if th is None else th
        s = self.s if s is None else s
        d = self.d if d is None else d
        p = self.probs(th[self.e] - s[self.r] - d[self.j], self.j, tau)
        ex = p @ KS
        return p, ex, np.maximum(p @ (KS ** 2) - ex ** 2, 1e-9)

    def fit(self, max_iter=1500, tol=1e-5):
        e, r, j, x = self.e, self.r, self.j, self.x
        for it in range(max_iter):
            _, ex, v = self.moments()
            self.th += np.clip(np.bincount(e, x - ex, self.nE) / np.maximum(np.bincount(e, v, self.nE), 1e-9), -1.0, 1.0)
            _, ex, v = self.moments()
            ds = np.clip(-np.bincount(r, x - ex, self.nR) / np.maximum(np.bincount(r, v, self.nR), 1e-9), -1.0, 1.0)
            self.s += ds
            _, ex, v = self.moments()
            dd = np.clip(-np.bincount(j, x - ex, self.nJ) / np.maximum(np.bincount(j, v, self.nJ), 1e-9), -1.0, 1.0)
            self.d += dd
            p, _, _ = self.moments()
            ge = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]             # P(X >= k), k = 0..4
            dt = np.zeros_like(self.tau)
            for jj in range(self.nJ):                               # Newton penuh 4 x 4 untuk ambang tiap kriteria
                m = j == jj
                A = ge[m][:, 1:]
                grad = (A - (x[m][:, None] >= np.arange(1, KC)[None, :])).sum(0)
                info = np.array([[(A[:, max(k, l)] - A[:, k] * A[:, l]).sum() for l in range(KC - 1)] for k in range(KC - 1)])
                dt[jj] = np.clip(np.linalg.solve(info + 1e-8 * np.eye(KC - 1), grad), -1.0, 1.0)
            self.tau += dt
            c = self.tau.mean(1, keepdims=True)                     # jumlah ambang tiap kriteria = 0; sisanya masuk ke d_j
            self.tau -= c
            self.d += c[:, 0]
            c = self.s.mean()                                       # rerata s = 0 dan rerata d = 0 (theta ikut bergeser)
            self.s -= c
            self.th -= c
            c = self.d.mean()
            self.d -= c
            self.th -= c
            step = max(np.abs(ds).max(), np.abs(dd).max(), np.abs(dt).max())
            if step < tol:
                break
        self.iters, self.converged = it + 1, bool(step < tol)
        return self

    def loglik(self, th=None, s=None, d=None, tau=None):
        """Log-likelihood bersama, dihitung langsung dari rumus: z_k = k * eta - jumlah ambang sampai k."""
        th = self.th if th is None else th
        s = self.s if s is None else s
        d = self.d if d is None else d
        z = (th[self.e] - s[self.r] - d[self.j])[:, None] * KS[None, :] - self.cum(tau)[self.j]
        mx = z.max(1)
        return float((z[np.arange(len(self.x)), self.x] - mx - np.log(np.exp(z - mx[:, None]).sum(1))).sum())

    def gradien(self, th=None, s=None, d=None, tau=None):
        """Turunan analitik log-likelihood terhadap theta, s, d, dan tau (persamaan skor yang dipakai fit)."""
        p, ex, _ = self.moments(th, s, d, tau)
        res = self.x - ex
        ge = np.cumsum(p[:, ::-1], axis=1)[:, ::-1][:, 1:]
        g_tau = np.array([(ge[self.j == jj] - (self.x[self.j == jj][:, None] >= np.arange(1, KC)[None, :])).sum(0)
                          for jj in range(self.nJ)])
        return (np.bincount(self.e, res, self.nE), -np.bincount(self.r, res, self.nR), -np.bincount(self.j, res, self.nJ), g_tau)

    def stat_faset(self, idx, n):
        """Galat baku (pendekatan informasi FACETS), infit, outfit, dan banyak pengamatan tiap elemen satu fasilitas."""
        _, ex, v = self.moments()
        res = self.x - ex
        hv, cnt = np.bincount(idx, v, n), np.bincount(idx, minlength=n)
        return dict(n_obs=cnt, se=1.0 / np.sqrt(np.maximum(hv, 1e-9)), infit=np.bincount(idx, res ** 2, n) / np.maximum(hv, 1e-9),
                    outfit=np.bincount(idx, res ** 2 / v, n) / np.maximum(cnt, 1))

    def geser_skor(self):
        """Selisih skor harapan (poin) tiap penilai dibanding penilai berketatan 0, pada esai berkemampuan rata-rata, rata-rata atas
        kriteria; positif = skor harapan lebih rendah (arah sama dengan keketatan)."""
        th0 = self.th.mean()
        out = np.zeros(self.nR)
        for rr in range(self.nR):
            for jj in range(self.nJ):
                jv = np.full(1, jj)
                p1 = self.probs(np.array([th0 - self.s[rr] - self.d[jj]]), jv)
                p0 = self.probs(np.array([th0 - self.d[jj]]), jv)
                out[rr] += (p0 @ KS - p1 @ KS)[0] / self.nJ
        return out

    def sel_bias(self):
        """Interaksi penilai x kriteria: bias = -(jumlah residu) / (jumlah varians); positif = lebih ketat daripada harapan model."""
        _, ex, v = self.moments()
        key, n = self.r * self.nJ + self.j, self.nR * self.nJ
        sr, sv, cnt = np.bincount(key, self.x - ex, n), np.bincount(key, v, n), np.bincount(key, minlength=n)
        return dict(n_obs=cnt.reshape(self.nR, self.nJ), bias=(-sr / np.maximum(sv, 1e-9)).reshape(self.nR, self.nJ),
                    se=(1.0 / np.sqrt(np.maximum(sv, 1e-9))).reshape(self.nR, self.nJ))

    def simulasi(self, rng):
        p = self.probs(self.th[self.e] - self.s[self.r] - self.d[self.j], self.j)
        return (rng.random(len(self.x))[:, None] > np.cumsum(p, 1)).sum(1).clip(0, KC - 1)


def pemisahan(est, se):
    """Indeks pemisahan G = SD terkoreksi / RMSE, keandalan R = G^2 / (1 + G^2), strata, dan khi-kuadrat tetap (homogenitas)."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    var_obs, rmse2 = est.var(ddof=1), np.mean(se ** 2)
    G = math.sqrt(max(var_obs - rmse2, 0.0) / rmse2)
    w = 1 / se ** 2
    chi = float((w * (est - (w * est).sum() / w.sum()) ** 2).sum())
    return dict(G=G, R=G ** 2 / (1 + G ** 2), strata=(4 * G + 1) / 3, chi2=chi, df=len(est) - 1)


def bh(p):
    """Penyesuaian Benjamini-Hochberg (nilai q)."""
    p = np.asarray(p, float)
    n, o = len(p), np.argsort(p)
    q, jalan = np.empty(n), 1.0
    for rank in range(n, 0, -1):
        i = o[rank - 1]
        jalan = min(jalan, p[i] * n / rank)
        q[i] = jalan
    return q


# ----------------------------------------------------------------------------------------------------------
# 3. Analisis satu versi: penilai, aspek, rasio, pembanding linear, bias, pemulihan parameter
# ----------------------------------------------------------------------------------------------------------
def linear_penilai(df, kriteria):
    """Penaksir linear (pemeriksaan silang keketatan, Kim 2026): selisih rerata skor slot 1 dan 2 dalam esai = b_r1 - b_r2 + galat,
    kuadrat terkecil norma minimal; keketatan = -b (poin skor, positif = lebih ketat). Mengembalikan Series menurut nama penilai."""
    penilai = sorted(set(df.Rater_1) | set(df.Rater_2), key=no)
    pi = {r: i for i, r in enumerate(penilai)}
    A = np.column_stack([df[f"{c}_1"].values for c in kriteria]).astype(float)
    Bm = np.column_stack([df[f"{c}_2"].values for c in kriteria]).astype(float)
    ok = (A > 0) & (Bm > 0)
    keep = ok.sum(1) > 0
    y = (np.where(ok, A - Bm, 0).sum(1) / np.maximum(ok.sum(1), 1))[keep]
    ia, ib = df.Rater_1.map(pi).values[keep], df.Rater_2.map(pi).values[keep]
    n = len(y)
    X = np.zeros((n, len(penilai)))
    X[np.arange(n), ia] += 1.0
    X[np.arange(n), ib] -= 1.0
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    beta -= beta.mean()
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (n - 1) / (n - np.linalg.matrix_rank(X))
    return pd.Series(-beta, index=penilai), float(adj)


def linear_kriteria(df, kriteria):
    """Kesulitan tanpa model (poin skor): skor tiap kriteria dikurangi rerata semua kriteria dari penilai dan esai yang sama,
    dirata-ratakan; dikali -1 agar positif = skor lebih rendah daripada kriteria lain. Penilai dan esai tersingkir lewat pengurangan."""
    M = np.vstack([m[(m > 0).all(1)] for m in
                   (np.column_stack([df[f"{c}_{s}"].values for c in kriteria]).astype(float) for s in (1, 2))])
    return -(M - M.mean(1, keepdims=True)).mean(0)


def ukuran_sebaran(v):
    v = np.asarray(v, float)
    return float(v.max() - v.min()), float(v.std(ddof=1))


def analisis(df, kode, rep=0, bias=True, verbose=True):
    kriteria = VERSI[kode]
    sp = siapkan(df, kriteria)
    e, r, j, x, penilai = sp["e"], sp["r"], sp["j"], sp["x"], sp["penilai"]
    n_r, n_j = sp["n_r"], sp["n_j"]
    m = MFRM(e, r, j, x, sp["n_e"], n_r, n_j).fit()
    if verbose:
        print(f"  JMLE: {m.iters} iterasi, konvergen = {m.converged}; {len(x)} pengamatan, {sp['n_e']} esai, {n_r} penilai, {n_j} kriteria")
    sr, sj = m.stat_faset(r, n_r), m.stat_faset(j, n_j)
    geser = m.geser_skor()
    lin_s_all, adjR2 = linear_penilai(df, kriteria)
    lin_s = lin_s_all.reindex(penilai).values
    lin_d = linear_kriteria(df, kriteria)
    pr = np.unique(r * sp["n_e"] + e)
    n_esai = np.bincount(pr // sp["n_e"], minlength=n_r)
    fit_ok = lambda t, rg: ((t.infit >= rg[0]) & (t.infit <= rg[1]) & (t.outfit >= rg[0]) & (t.outfit <= rg[1]))
    tr = pd.DataFrame(dict(penilai=penilai, n_esai=n_esai, n_pengamatan=sr["n_obs"], keketatan_logit=m.s, se=sr["se"],
                           ci_lo=m.s - 1.96 * sr["se"], ci_hi=m.s + 1.96 * sr["se"], infit=sr["infit"], outfit=sr["outfit"],
                           selisih_skor_poin=geser, keketatan_linear_poin=lin_s))
    tr["cocok_kim"] = fit_ok(tr, FIT_KIM)
    tr["cocok_wu"] = fit_ok(tr, FIT_WU)
    tj = pd.DataFrame(dict(kriteria=kriteria, kesulitan_logit=m.d, se=sj["se"], infit=sj["infit"], outfit=sj["outfit"],
                           kesulitan_linear_poin=lin_d))
    for k in range(KC - 1):
        tj[f"ambang_{k + 1}"] = m.tau[:, k]
    tj["cocok_kim"] = fit_ok(tj, FIT_KIM)
    tj["cocok_wu"] = fit_ok(tj, FIT_WU)
    # ukuran utama: rentang dan simpangan baku keketatan penilai vs kesulitan aspek (Kim membandingkan rentangnya)
    rs, ss = ukuran_sebaran(m.s)
    rd, sd = ukuran_sebaran(m.d)
    besar = tr.n_esai >= MIN_ESAI
    rs100 = ukuran_sebaran(m.s[besar.values])[0]
    rls, _ = ukuran_sebaran(lin_s)
    rld, _ = ukuran_sebaran(lin_d)
    sep_s, sep_d = pemisahan(m.s, sr["se"]), pemisahan(m.d, sj["se"])
    # ketidakpastian dari galat baku: undian normal tiap elemen (hanya galat estimasi, bukan pemilihan penilai/aspek)
    rng = np.random.default_rng(SEED)
    S = m.s + sr["se"] * rng.standard_normal((B_MC, n_r))
    D = m.d + sj["se"] * rng.standard_normal((B_MC, n_j))
    mc_s, mc_d = S.max(1) - S.min(1), D.max(1) - D.min(1)
    ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
    sr_rank = pd.Series(m.s).rank().values
    rho = float(np.corrcoef(sr_rank, pd.Series(lin_s).rank().values)[0, 1])
    kat = np.bincount(j * KC + x, minlength=n_j * KC).reshape(n_j, KC)
    ring = dict(kode=kode, versi=NAMA_VERSI[kode],
                n_pengamatan=len(x), n_esai=sp["n_e"], n_esai_ekstrem=sp["n_ekstrem"], n_penilai=n_r, n_kriteria=n_j,
                pengamatan_per_esai=len(x) / sp["n_e"], n_komponen=sp["n_komp"], iterasi=m.iters, konvergen=m.converged,
                rentang_s=rs, sd_s=ss, rentang_d=rd, sd_d=sd, rasio_rentang=rs / rd, rasio_sd=ss / sd, rasio_balik=rd / rs,
                rentang_s_lo=ci(mc_s)[0], rentang_s_hi=ci(mc_s)[1], rentang_d_lo=ci(mc_d)[0], rentang_d_hi=ci(mc_d)[1],
                rasio_lo=ci(mc_s / mc_d)[0], rasio_hi=ci(mc_s / mc_d)[1],
                n_penilai_min_esai=int(besar.sum()), rentang_s_min_esai=rs100, rasio_rentang_min_esai=rs100 / rd,
                G_s=sep_s["G"], R_s=sep_s["R"], chi2_s=sep_s["chi2"], df_s=sep_s["df"], G_d=sep_d["G"], R_d=sep_d["R"],
                chi2_d=sep_d["chi2"], df_d=sep_d["df"],
                n_luar_kim_s=int((~tr.cocok_kim).sum()), n_luar_wu_s=int((~tr.cocok_wu).sum()),
                n_luar_kim_d=int((~tj.cocok_kim).sum()), n_luar_wu_d=int((~tj.cocok_wu).sum()),
                r_pearson=float(np.corrcoef(m.s, lin_s)[0, 1]), r_spearman=rho, adjR2_identitas=adjR2,
                r_pearson_d=float(np.corrcoef(m.d, lin_d)[0, 1]), kat_min=int(kat.min()),
                rentang_linear_s=rls, rentang_linear_d=rld, rasio_linear=rls / rld,
                geser_min=float(geser.min()), geser_maks=float(geser.max()), rentang_geser=float(geser.max() - geser.min()))
    tb = None
    if bias:
        bc = m.sel_bias()
        rows = [dict(penilai=penilai[a], kriteria=kriteria[b], n_pengamatan=int(bc["n_obs"][a, b]), bias_logit=float(bc["bias"][a, b]),
                     se=float(bc["se"][a, b])) for a in range(n_r) for b in range(n_j)]
        tb = pd.DataFrame(rows)
        tb = tb[tb.n_pengamatan >= BIAS_MIN_OBS].reset_index(drop=True)
        tb["t"] = tb.bias_logit / tb.se
        tb["p"] = [math.erfc(abs(t) / math.sqrt(2)) for t in tb.t]            # dua sisi, hampiran normal
        tb["q_bh"] = bh(tb.p.values)
        tb["tanda_wu"] = (tb.t.abs() > BIAS_T) & (tb.bias_logit.abs() > BIAS_ABS)
        tb["tanda_wu_dan_bh"] = tb.tanda_wu & (tb.q_bh < 0.05)
        tb["arah"] = np.where(tb.bias_logit > 0, "lebih ketat", "lebih longgar")
        ring.update(bias_n_sel=len(tb), bias_n_tanda=int(tb.tanda_wu.sum()), bias_n_tanda_bh=int(tb.tanda_wu_dan_bh.sum()),
                    bias_n_bh05=int((tb.q_bh < 0.05).sum()), bias_maks_abs=float(tb.bias_logit.abs().max()))
    pem = None
    if rep > 0:
        rg = np.random.default_rng(SEED)
        h = []
        for _ in range(rep):                        # data dibangkitkan dari parameter hasil estimasi, lalu diestimasi ulang
            ms = MFRM(e, r, j, m.simulasi(rg), sp["n_e"], n_r, n_j)
            ms.th, ms.s, ms.d, ms.tau = m.th.copy(), m.s.copy(), m.d.copy(), m.tau.copy()
            ms.fit(tol=1e-4)
            h.append((np.corrcoef(ms.s, m.s)[0, 1], np.polyfit(m.s, ms.s, 1)[0], np.polyfit(m.d, ms.d, 1)[0],
                      (np.ptp(ms.s) / np.ptp(ms.d)) / (np.ptp(m.s) / np.ptp(m.d))))
        h = np.array(h).mean(0)
        pem = dict(n_rep=rep, r_s=float(h[0]), kemiringan_s=float(h[1]), kemiringan_d=float(h[2]), rasio_pulih=float(h[3]),
                   teori=float(len(x) / sp["n_e"] / (len(x) / sp["n_e"] - 1)))
    return dict(m=m, sp=sp, tr=tr, tj=tj, tb=tb, ring=ring, pem=pem, kriteria=kriteria)


# ----------------------------------------------------------------------------------------------------------
# 4. Pemeriksaan: hitung ulang dengan cara lain dan bandingkan
# ----------------------------------------------------------------------------------------------------------
def periksa(h, df):
    """Pemeriksaan satu versi; hentikan program bila ada yang tidak lolos. Tiap baris: nama, nilai, batas (lolos bila nilai < batas)."""
    m, sp, kriteria = h["m"], h["sp"], h["kriteria"]
    rng = np.random.default_rng(SEED + 1)
    cek = []
    # 1. probabilitas kategori sama dengan rumus langkah berdekatan yang dihitung manual
    idx = rng.choice(len(m.x), 300, replace=False)
    eta = (m.th[m.e] - m.s[m.r] - m.d[m.j])[idx]
    P = m.probs(eta, m.j[idx])
    dif = 0.0
    for i, k in enumerate(idx):
        w = [1.0]
        for t in range(1, KC):
            w.append(w[-1] * math.exp(eta[i] - m.tau[m.j[k], t - 1]))
        dif = max(dif, max(abs(P[i, t] - w[t] / sum(w)) for t in range(KC)))
    cek.append(("Probabilitas kategori = rumus langkah berdekatan, hitungan manual (300 pengamatan)", dif, 1e-12))
    # 2. hitungan banyak pengamatan dan esai dari jalur lain
    n_lain = sum(int((df[f"{c}_{s}"] > 0).sum()) for c in kriteria for s in (1, 2))
    cek.append(("Pengamatan awal = hitungan ulang langsung dari kolom skor", abs(sp["n_awal"] - n_lain), 0.5))
    cek.append(("Pengamatan dipakai = awal - esai ekstrem - di luar komponen terbesar",
                abs(len(m.x) - (sp["n_awal"] - sp["n_ekstrem_obs"] - sp["n_dibuang_komp"])), 0.5))
    cek.append(("Esai dipakai = esai unik (Filename) - esai ekstrem", abs(m.nE - (df.Filename.nunique() - sp["n_ekstrem"])), 0.5))
    cek.append(("Semua penilai terhubung (satu komponen)", sp["n_komp"] - 1, 0.5))
    cek.append(("Estimasi konvergen", 0.0 if m.converged else 1.0, 0.5))
    # 3. turunan analitik sama dengan turunan numerik log-likelihood, di titik acak (turunan besar, bukan di titik optimum)
    th0, s0, d0 = (m.th + 0.3 * rng.standard_normal(m.nE), m.s + 0.3 * rng.standard_normal(m.nR), m.d + 0.3 * rng.standard_normal(m.nJ))
    t0 = m.tau + 0.3 * rng.standard_normal(m.tau.shape)
    g_th, g_s, g_d, g_t = m.gradien(th0, s0, d0, t0)
    eps, dif = 1e-5, 0.0

    def numerik(fungsi):
        return (fungsi(eps) - fungsi(-eps)) / (2 * eps)

    for i in rng.choice(m.nE, 5, replace=False):
        def fth(a, i=i):
            t = th0.copy(); t[i] += a
            return m.loglik(t, s0, d0, t0)
        dif = max(dif, abs(numerik(fth) - g_th[i]) / max(1, abs(g_th[i])))
    for i in range(m.nR):
        def fs(a, i=i):
            t = s0.copy(); t[i] += a
            return m.loglik(th0, t, d0, t0)
        dif = max(dif, abs(numerik(fs) - g_s[i]) / max(1, abs(g_s[i])))
    for i in range(m.nJ):
        def fd(a, i=i):
            t = d0.copy(); t[i] += a
            return m.loglik(th0, s0, t, t0)
        dif = max(dif, abs(numerik(fd) - g_d[i]) / max(1, abs(g_d[i])))
        for k in range(KC - 1):
            def ft(a, i=i, k=k):
                t = t0.copy(); t[i, k] += a
                return m.loglik(th0, s0, d0, t)
            dif = max(dif, abs(numerik(ft) - g_t[i, k]) / max(1, abs(g_t[i, k])))
    cek.append(("Turunan analitik = turunan numerik log-likelihood (theta, s, d, tau; selisih relatif maks)", dif, 1e-5))
    # 4. di hasil estimasi semua turunan harus nol (titik stasioner log-likelihood bersama)
    g_th, g_s, g_d, g_t = m.gradien()
    _, _, v = m.moments()
    cek.append(("Persamaan skor esai: langkah Newton maks (logit)", np.abs(g_th / np.bincount(m.e, v, m.nE)).max(), 1e-3))
    cek.append(("Persamaan skor penilai: langkah Newton maks (logit)", np.abs(g_s / np.bincount(m.r, v, m.nR)).max(), 1e-4))
    cek.append(("Persamaan skor kriteria: langkah Newton maks (logit)", np.abs(g_d / np.bincount(m.j, v, m.nJ)).max(), 1e-4))
    cek.append(("Persamaan skor ambang: turunan maks per pengamatan kriteria",
                max(np.abs(g_t[jj]).max() / (m.j == jj).sum() for jj in range(m.nJ)), 1e-4))
    # 5. identifikasi
    cek.append(("Rerata keketatan = 0", abs(m.s.mean()), 1e-9))
    cek.append(("Rerata kesulitan = 0", abs(m.d.mean()), 1e-9))
    cek.append(("Jumlah ambang tiap kriteria = 0", np.abs(m.tau.sum(1)).max(), 1e-9))
    # 6. rumus keandalan dari pemisahan sama dengan pasangan angka Kim (2026): G = 7,33 -> R = 0,982
    cek.append(("Rumus keandalan R = G^2/(1+G^2) pada G Kim = 7,33 memberi 0,982", abs(7.33 ** 2 / (1 + 7.33 ** 2) - 0.982), 5e-4))
    ok = True
    for nama, nilai, batas in cek:
        lolos = bool(nilai < batas)
        ok &= lolos
        print(f"  {'LOLOS' if lolos else 'GAGAL'}  {nama}: {nilai:.3g} (batas {batas:g})")
    if not ok:
        raise SystemExit("Pemeriksaan gagal.")


# ----------------------------------------------------------------------------------------------------------
# 5. Sensitivitas, keluaran, dan program utama
# ----------------------------------------------------------------------------------------------------------
def sensitivitas(df, utama):
    """Ulangi pada tiga himpunan baris (himpunan utama lebih dulu, dipakai ulang dari hasil utama) dari semua baris."""
    if "text_id_kaggle" not in df.columns:
        return None
    baris = []
    for kunci in [UTAMA] + [k for k in HIMPUNAN if k != UTAMA]:
        nama = HIMPUNAN[kunci] + (" (utama)" if kunci == UTAMA else "")
        d = df[mask_himpunan(df, kunci)].reset_index(drop=True)
        for kode in VERSI:
            print(f"  {nama}, {NAMA_VERSI[kode]}")
            ring = dict(utama[kode]) if kunci == UTAMA else analisis(d, kode, rep=0, bias=False, verbose=False)["ring"]
            baris.append(dict(himpunan=nama, n_baris=len(d), **ring))
    return pd.DataFrame(baris)


def f(x, nd=2):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}".replace(".", ",")


def fi(n):
    return f"{int(n):,}".replace(",", ".")


def md(kepala, baris):
    return "\n".join(["| " + " | ".join(kepala) + " |", "|" + "---|" * len(kepala)] + ["| " + " | ".join(r) + " |" for r in baris]) + "\n"


def teks_versi(h):
    g, tr, tj, tb, pem = h["ring"], h["tr"], h["tj"], h["tb"], h["pem"]
    s = [f"## Versi: {g['versi']}\n",
         f"{fi(g['n_pengamatan'])} pengamatan dari {fi(g['n_esai'])} esai (esai skor ekstrem dikeluarkan: {g['n_esai_ekstrem']}), "
         f"{g['n_penilai']} penilai, {g['n_kriteria']} kriteria; {f(g['pengamatan_per_esai'], 1)} pengamatan per esai. "
         f"Konvergen: {'ya' if g['konvergen'] else 'TIDAK'} ({g['iterasi']} iterasi). Komponen terhubung: {g['n_komponen']}.\n",
         "### Ukuran utama\n",
         md(["Ukuran", "Penilai (keketatan)", "Kriteria (kesulitan)"],
            [["Banyak elemen", str(g["n_penilai"]), str(g["n_kriteria"])],
             ["Rentang (logit) [galat estimasi, CI 95%]", f"{f(g['rentang_s'])} [{f(g['rentang_s_lo'])}; {f(g['rentang_s_hi'])}]",
              f"{f(g['rentang_d'])} [{f(g['rentang_d_lo'])}; {f(g['rentang_d_hi'])}]"],
             ["Simpangan baku (logit)", f(g["sd_s"]), f(g["sd_d"])],
             ["Indeks pemisahan G", f(g["G_s"]), f(g["G_d"])],
             ["Keandalan pemisahan R", f(g["R_s"], 3), f(g["R_d"], 3)],
             ["Khi-kuadrat tetap (db)", f"{f(g['chi2_s'], 0)} ({g['df_s']})", f"{f(g['chi2_d'], 0)} ({g['df_d']})"],
             [f"Di luar infit/outfit {f(FIT_KIM[0], 1)}-{f(FIT_KIM[1], 1)} (Kim)", f"{g['n_luar_kim_s']} dari {g['n_penilai']}",
              f"{g['n_luar_kim_d']} dari {g['n_kriteria']}"],
             [f"Di luar infit/outfit {f(FIT_WU[0], 1)}-{f(FIT_WU[1], 1)} (Wu dkk.)", f"{g['n_luar_wu_s']} dari {g['n_penilai']}",
              f"{g['n_luar_wu_d']} dari {g['n_kriteria']}"]]),
         f"Rasio rentang penilai : kriteria = {f(g['rasio_rentang'], 2)} [{f(g['rasio_lo'], 2)}; {f(g['rasio_hi'], 2)}] "
         f"(kebalikannya, kriteria : penilai seperti cara Kim: {f(g['rasio_balik'], 2)}). Rasio simpangan baku = {f(g['rasio_sd'], 2)}. "
         f"Hanya penilai dengan ≥ {MIN_ESAI} esai ({g['n_penilai_min_esai']} penilai): rentang {f(g['rentang_s_min_esai'])}, "
         f"rasio {f(g['rasio_rentang_min_esai'], 2)}.\n",
         f"Pembanding tanpa model (poin skor): rentang keketatan linear penilai {f(g['rentang_linear_s'], 2)}, rentang kesulitan linear "
         f"kriteria {f(g['rentang_linear_d'], 2)}, rasio {f(g['rasio_linear'], 2)}. Korelasi keketatan MFRM dengan linear: Pearson "
         f"{f(g['r_pearson'], 3)}, peringkat {f(g['r_spearman'], 3)}; korelasi kesulitan MFRM dengan linear: {f(g['r_pearson_d'], 3)}. "
         f"Selisih skor harapan penilai terhadap penilai berketatan rata-rata, pada esai berkemampuan rata-rata (positif = skor lebih rendah): "
         f"{f(g['geser_min'])} sampai {f(g['geser_maks'])} poin (rentang {f(g['rentang_geser'])} poin). "
         f"Pengamatan paling sedikit pada satu kategori skor satu kriteria: {fi(g['kat_min'])}.\n",
         "### Keketatan tiap penilai (urut dari paling longgar)\n",
         md(["Penilai", "Esai", "Keketatan logit [CI 95%]", "Infit", "Outfit", "Selisih skor (poin)", "Linear (poin)"],
            [[r.penilai, fi(r.n_esai), f"{f(r.keketatan_logit)} [{f(r.ci_lo)}; {f(r.ci_hi)}]", f(r.infit), f(r.outfit),
              f(r.selisih_skor_poin), f(r.keketatan_linear_poin)] for _, r in tr.sort_values("keketatan_logit").iterrows()]),
         "### Kesulitan tiap kriteria (urut dari paling mudah)\n",
         md(["Kriteria", "Kesulitan logit", "SE", "Infit", "Outfit", "Linear (poin)"],
            [[r.kriteria, f(r.kesulitan_logit), f(r.se, 3), f(r.infit), f(r.outfit), f(r.kesulitan_linear_poin, 3)]
             for _, r in tj.sort_values("kesulitan_logit").iterrows()])]
    if tb is not None:
        s += [f"### Bias penilai x kriteria (Wu dkk.: tanda bila |t| > {f(BIAS_T, 1)} dan |bias| > {f(BIAS_ABS, 1)} logit)\n",
              f"{g['bias_n_sel']} sel dengan ≥ {BIAS_MIN_OBS} pengamatan. Bertanda: {g['bias_n_tanda']} sel; tetap bertanda setelah "
              f"Benjamini-Hochberg (q < 0,05): {g['bias_n_tanda_bh']} sel. Sel dengan q < 0,05 tanpa syarat ukuran: {g['bias_n_bh05']}. "
              f"|Bias| terbesar: {f(g['bias_maks_abs'])} logit.\n"]
        tt = tb[tb.tanda_wu_dan_bh]
        if len(tt):
            tab = tt.groupby(["kriteria", "arah"]).size().unstack(fill_value=0).reindex(h["kriteria"]).fillna(0).astype(int)
            s.append(md(["Kriteria"] + list(tab.columns), [[k] + [str(v) for v in r.values] for k, r in tab.iterrows()]))
            s.append(md(["Penilai", "Kriteria", "Pengamatan", "Bias (logit)", "t", "Arah"],
                        [[r.penilai, r.kriteria, str(r.n_pengamatan), f(r.bias_logit), f(r.t, 1), r.arah]
                         for _, r in tt.reindex(tt.t.abs().sort_values(ascending=False).index).head(10).iterrows()]))
    if pem is not None:
        s += ["### Pemeriksaan estimasi (pemulihan parameter)\n",
              f"Data dibangkitkan dari parameter hasil estimasi lalu diestimasi ulang ({pem['n_rep']} ulangan): korelasi keketatan "
              f"{f(pem['r_s'], 3)}; kemiringan keketatan estimasi terhadap sebenarnya {f(pem['kemiringan_s'], 3)}; kemiringan kesulitan "
              f"{f(pem['kemiringan_d'], 3)}; rasio rentang hasil ulang dibanding rasio rentang sebenarnya {f(pem['rasio_pulih'], 3)}. "
              f"Perkiraan teori pembesaran JMLE L/(L-1) = {f(pem['teori'], 3)} dengan L = {f(g['pengamatan_per_esai'], 1)} pengamatan per esai.\n"]
    return "\n".join(s)


def ringkasan(hasil, sens):
    a_min, a_max = KIM["analitik_min"], KIM["analitik_maks"]
    rt = a_max - a_min
    s = ["# RQ1c: keketatan penilai dibanding kesulitan aspek\n",
         "Model many-facet Rasch kredit parsial, JMLE. Keketatan positif = lebih ketat; kesulitan positif = lebih sulit. "
         "Rasio dihitung dari rentang (seperti Kim) dan dari simpangan baku.\n",
         f"Himpunan utama: {HIMPUNAN[UTAMA]}.\n",
         "## Konteks: Kim (2026), bukan hasil kita\n",
         f"Rentang keketatan penilai {f(KIM['rentang_penilai'], 3)} logit (42 penilai), pemisahan {f(KIM['G_penilai'])}, keandalan "
         f"{f(KIM['R_penilai'], 3)}; rentang kesulitan 9 kriteria sekitar {f(KIM['rentang_kriteria'])} logit, dengan kriteria holistik "
         f"(+{f(KIM['holistik'], 3)}) sebagai yang tersulit. Tanpa holistik, rentang 8 kriteria analitik = {f(a_max, 3)} - ({f(a_min, 3)}) = "
         f"{f(rt, 3)}; rentang penilai : rentang analitik = {f(KIM['rentang_penilai'] / rt, 2)} (hitungan kami dari Tabel 9 Kim, bukan klaim Kim).\n"]
    for kode in VERSI:
        s.append(teks_versi(hasil[kode]))
    if sens is not None:
        s.append("## Sensitivitas menurut himpunan baris\n")
        s.append(md(["Himpunan", "Versi", "Baris", "Rentang penilai (logit)", "Rentang kriteria (logit)", "Rasio rentang", "Rasio SD",
                     "Rentang selisih skor penilai (poin)"],
                    [[r.himpunan, r.versi, fi(r.n_baris), f(r.rentang_s), f(r.rentang_d), f(r.rasio_rentang),
                      f(r.rasio_sd), f(r.rentang_geser)] for _, r in sens.iterrows()]))
        s.append("Logit dari dua himpunan yang berbeda tidak sebanding langsung (ambang dan skala diestimasi terpisah di tiap himpunan); "
                 "rasio dan poin skor sebanding.\n")
    return "\n".join(s)


def main(csv=None, keluar=None, rep=3):
    csv = csv or os.path.join(HERE, "data", "ellipse_raw.csv")
    keluar = keluar or os.path.join(HERE, "hasil_rq1c")
    os.makedirs(keluar, exist_ok=True)
    df_semua = muat(csv)
    df = pilih_utama(df_semua)
    hasil = {}
    for kode in VERSI:
        print(f"\nVersi {NAMA_VERSI[kode]}:")
        hasil[kode] = analisis(df, kode, rep=rep)
    for kode in VERSI:
        print(f"\nPemeriksaan hitungan, versi {NAMA_VERSI[kode]}:")
        periksa(hasil[kode], df)
    print("\nSensitivitas menurut himpunan baris:")
    sens = sensitivitas(df_semua, {k: h["ring"] for k, h in hasil.items()})
    for kode, h in hasil.items():
        h["tr"].to_csv(os.path.join(keluar, f"rq1c_penilai_{kode}.csv"), index=False)
        h["tj"].to_csv(os.path.join(keluar, f"rq1c_kriteria_{kode}.csv"), index=False)
        h["tb"].to_csv(os.path.join(keluar, f"rq1c_bias_{kode}.csv"), index=False)
    pd.DataFrame([h["ring"] for h in hasil.values()]).to_csv(os.path.join(keluar, "rq1c_ringkasan_versi.csv"), index=False)
    if sens is not None:
        sens.to_csv(os.path.join(keluar, "rq1c_himpunan_bagian.csv"), index=False)
    teks = ringkasan(hasil, sens)
    with open(os.path.join(keluar, "ringkasan_rq1c.md"), "w", encoding="utf-8") as fh:
        fh.write(teks)
    print("\n" + teks)
    print(f"Selesai. Berkas ada di: {keluar}")


if __name__ == "__main__":
    argv, rep = sys.argv[1:], 3
    if "--rep" in argv:
        i = argv.index("--rep")
        rep = int(argv[i + 1])
        del argv[i:i + 2]
    if "--utama" in argv:
        i = argv.index("--utama")
        UTAMA = argv[i + 1] if i + 1 < len(argv) else ""
        del argv[i:i + 2]
        if UTAMA not in ("resmi", "semua"):
            raise SystemExit("--utama harus resmi atau semua")
    main(argv[0] if len(argv) > 0 else None, argv[1] if len(argv) > 1 else None, rep)
