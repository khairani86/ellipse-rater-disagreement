"""common.py - data, lipatan per seed, fungsi skor, dan ukuran yang dipakai bersama oleh seluruh paket RQ2.

Folder keluaran: work/ (token), preds/ (prediksi di luar lipatan), hasil/ (tabel dan laporan kontrol).
Variabel lingkungan RQ2_TAG=_x mengganti nama folder menjadi work_x/, preds_x/, hasil_x/ (dipakai uji_mandiri.py).

Data: berkas mentah ELLIPSE, HIMPUNAN RESMI saja (baris ber-text_id_kaggle dengan Overall > 0 pada kedua penilai).
Satu baris = satu esai, jadi pembagian lipatan menurut baris sama dengan pembagian menurut esai.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

for _st in (sys.stdout, sys.stderr):                 # konsol Windows: karakter khusus diganti '?', jangan berhenti
    try:
        _st.reconfigure(errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get("RQ2_TAG", "")
METODE = ["K2", "K3", "K4", "K5", "K6"]
PEMBANDING = ["B0", "B1", "B2", "B3"]
NAMA = {
    "K2": "Rerata dua penilai (pembanding bagi K3 sampai K6)",
    "K3": "Skor tertinggi dari dua penilai",
    "K4": "Skor terendah dari dua penilai",
    "K5": "Rerata, esai selisih >= 2 tidak dipakai melatih",
    "K6": "Skor harapan model Rasch keketatan penilai",
    "B0": "Konstanta: rerata skor latih",
    "B1": "Ridge pada TF-IDF (kata 1-2 gram)",
    "B2": "BERT standar, kepala regresi, sasaran K2",
    "B3": "DeBERTa-v3-base, kepala regresi, sasaran K2",
}
SKALA = np.arange(1, 6)


def path(*p):
    return os.path.join(ROOT, *p)


def wpath(top, *p):
    """top salah satu dari 'work', 'preds', 'hasil'; mengikuti RQ2_TAG."""
    return os.path.join(ROOT, top + TAG, *p)


def config():
    with open(path("config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if os.environ.get("RQ2_DEBERTA"):                # hanya untuk uji dengan model mini lokal
        cfg["model_deberta"] = os.environ["RQ2_DEBERTA"]
    if os.environ.get("RQ2_BERT"):
        cfg["model_bert"] = os.environ["RQ2_BERT"]
    return cfg


def muat_encoder(nama):
    """Muat encoder dalam fp32. transformers >= 5 memuat bobot dalam dtype checkpoint secara bawaan, dan checkpoint
    deberta-v3-base disimpan dalam fp16; pelatihan fp16 murni membuat model runtuh (semua prediksi sama).
    bf16 dipakai hanya lewat autocast, sedangkan bobot dan keadaan optimizer tetap fp32."""
    import torch
    from transformers import AutoModel
    try:
        enc = AutoModel.from_pretrained(nama, dtype=torch.float32)
    except TypeError:                                  # transformers lama memakai nama torch_dtype
        enc = AutoModel.from_pretrained(nama, torch_dtype=torch.float32)
    bad = sorted({str(p.dtype) for p in enc.parameters() if p.dtype != torch.float32})
    if bad:
        raise SystemExit(f"bobot encoder bukan fp32 ({bad}); dihentikan")
    return enc


# ------------------------------------------------------------------ data dan lipatan
def muat_baris(csv=None):
    """Himpunan resmi: baris ber-text_id_kaggle, Overall > 0 pada kedua penilai. Kolom tambahan: row_id, x_a, x_b, d, rerata."""
    csv = csv or path("data", "ellipse_raw.csv")
    if not os.path.exists(csv):
        raise SystemExit(f"CSV tidak ditemukan: {csv}\nTaruh berkas ELLIPSE (skor mentah penilai) di data/ellipse_raw.csv")
    df = pd.read_csv(csv)
    perlu = ["text_id_kaggle", "Text", "Rater_1", "Rater_2", "Overall_1", "Overall_2"]
    hilang = [c for c in perlu if c not in df.columns]
    if hilang:
        raise SystemExit(f"Kolom tidak ada di CSV: {hilang}")
    n0 = len(df)
    df = df[df.text_id_kaggle.notna()]
    n_id = len(df)
    df = df[(df.Overall_1 > 0) & (df.Overall_2 > 0)].reset_index(drop=True)
    if not df.text_id_kaggle.is_unique:
        raise SystemExit("text_id_kaggle tidak unik: satu esai muncul lebih dari sekali")
    df["row_id"] = np.arange(len(df))
    df["x_a"] = df.Overall_1.astype(int)
    df["x_b"] = df.Overall_2.astype(int)
    df["d"] = (df.x_a - df.x_b).abs()
    df["rerata"] = (df.x_a + df.x_b) / 2
    df.attrs.update(n_berkas=n0, n_ber_id=n_id, n_dibuang_nol=n_id - len(df))
    return df


def rater_arrays(df):
    """Indeks penilai untuk slot 1 dan 2, dengan urutan penilai yang selalu sama (rater_1, rater_2, ...)."""
    def kunci(s):
        try:
            return (0, int(str(s).split("_")[-1]))
        except ValueError:
            return (1, str(s))
    raters = sorted(set(df.Rater_1) | set(df.Rater_2), key=kunci)
    ri = {r: i for i, r in enumerate(raters)}
    ia = np.array([ri[r] for r in df.Rater_1])
    ib = np.array([ri[r] for r in df.Rater_2])
    return raters, ia, ib, len(raters)


def lipatan(n, seed, cfg=None):
    """Pembagian 5 lipatan yang BARU untuk tiap seed (acak, tanpa lapisan): fold_seed + seed."""
    from sklearn.model_selection import KFold
    cfg = cfg or config()
    kf = KFold(cfg["n_folds"], shuffle=True, random_state=cfg["fold_seed"] + int(seed))
    f = np.full(n, -1)
    for k, (_, te) in enumerate(kf.split(np.arange(n))):
        f[te] = k
    return f


def bagi(n, seed, fold, cfg=None):
    """(indeks latih, indeks uji) untuk satu seed dan satu lipatan."""
    f = lipatan(n, seed, cfg)
    return np.where(f != fold)[0], np.where(f == fold)[0]


def semua_kombinasi(cfg=None):
    cfg = cfg or config()
    return [(s, k) for s in cfg["seeds"] for k in range(cfg["n_folds"])]


# ------------------------------------------------------------------ skor dan sebaran
def half_up(y):
    """Pembulatan ke bilangan bulat, 0,5 naik; hasil dipotong ke 1..5."""
    return np.clip(np.floor(np.asarray(y, float) + 0.5), 1, 5).astype(int)


def kisi_setengah(y):
    """Pembulatan ke kisi 0,5 (sembilan tingkat), 0,25 naik; hasil dipotong ke 1,0..5,0."""
    return np.clip(np.floor(2 * np.asarray(y, float) + 0.5) / 2, 1.0, 5.0)


def dist_dari_rerata(y):
    """Skor kontinu pada 1..5 -> sebaran atas DUA kelas bersebelahan dengan nilai harapan sama dengan y. (n, 5)."""
    y = np.clip(np.asarray(y, float), 1, 5)
    lo = np.minimum(np.floor(y + 1e-12).astype(int), 5)
    fr = y - lo
    fr[np.abs(fr) < 1e-12] = 0.0
    P = np.zeros((len(y), 5))
    n = np.arange(len(y))
    P[n, lo - 1] += 1 - fr
    P[n, np.minimum(lo + 1, 5) - 1] += fr
    return P


def satu_kelas(c):
    c = np.asarray(c, int)
    P = np.zeros((len(c), 5))
    P[np.arange(len(c)), c - 1] = 1.0
    return P


def nilai_harapan(P):
    return np.asarray(P) @ SKALA


# ------------------------------------------------------------------ ukuran kinerja
def qwk_label(a, b, K):
    """QWK dari dua vektor label bulat 0..K-1."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    O = np.bincount(a * K + b, minlength=K * K).reshape(K, K).astype(float)
    n = O.sum()
    E = O.sum(1)[:, None] * O.sum(0)[None, :] / n
    W = np.subtract.outer(np.arange(K), np.arange(K)) ** 2 / (K - 1) ** 2
    den = (W * E).sum()
    return float("nan") if den == 0 else float(1.0 - (W * O).sum() / den)


def qwk_skor(pred, acuan, kisi):
    """QWK antara prediksi (kontinu) dan skor acuan setelah keduanya dibulatkan ke kisi.
    kisi 'bulat'   : bilangan bulat 1..5 (5 tingkat), 0,5 naik, dipotong ke 1..5
    kisi 'setengah': kelipatan 0,5 dari 1,0 sampai 5,0 (9 tingkat), dipotong ke 1,0..5,0"""
    if kisi == "bulat":
        return qwk_label(half_up(pred) - 1, half_up(acuan) - 1, 5)
    if kisi == "setengah":
        ia = np.rint(2 * (kisi_setengah(pred) - 1)).astype(int)
        ib = np.rint(2 * (kisi_setengah(acuan) - 1)).astype(int)
        return qwk_label(ia, ib, 9)
    raise ValueError(kisi)


def prmse(M, h1, h2):
    """PRMSE (Loukina dkk., 2020, BEA, Lampiran B, persamaan 2-8, untuk dua penilai per esai):
    V = sum((h2 - h1)^2) / (2N); Var(T) = [2 sum(hbar - rata)^2 - (N - 1) V] / [2 (N - 1)];
    MSE(T|M) = rerata((hbar - M)^2) - V / 2; PRMSE = 1 - MSE / Var(T)."""
    M, h1, h2 = (np.asarray(v, float) for v in (M, h1, h2))
    n = len(M)
    hb = (h1 + h2) / 2
    V = np.sum((h2 - h1) ** 2) / (2 * n)
    var_t = (2 * np.sum((hb - hb.mean()) ** 2) - (n - 1) * V) / (2 * (n - 1))
    mse = np.mean((hb - M) ** 2) - V / 2
    return float(1.0 - mse / var_t)


def batas_kappa(a, b):
    """Batas kinerja menurut Uto (2026), seperti RQ1a: rho_1 = ICC(1,1), rho_Y = 2 rho_1 / (1 + rho_1);
    kappa_max = sqrt(rho_Y) (model sempurna); kappa_HL = sqrt(rho_1 rho_Y) (model yang galatnya sebesar satu penilai)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    m = (a + b) / 2
    msb = 2 * np.sum((m - m.mean()) ** 2) / (n - 1)
    msw = np.sum((a - b) ** 2) / 2 / n
    rho1 = (msb - msw) / (msb + msw)
    rho_y = (msb - msw) / msb
    return dict(rho1=float(rho1), rho_Y=float(rho_y), kappa_HL=float(np.sqrt(max(rho1 * rho_y, 0))),
                kappa_max=float(np.sqrt(max(rho_y, 0))))


# ------------------------------------------------------------------ inferensi
def nadeau_bengio(selisih, n_uji, n_latih):
    """Uji-t sampel berulang terkoreksi (Nadeau & Bengio, 2003): t = rerata / sqrt((1/J + n_uji/n_latih) var),
    derajat bebas J - 1. `selisih` = selisih berpasangan per (seed, lipatan). Mengembalikan (t, p dua sisi, faktor SE)."""
    from scipy import stats
    d = np.asarray(selisih, float)
    J = len(d)
    v = d.var(ddof=1)
    faktor = np.sqrt(1 + J * n_uji / n_latih)           # kenaikan kesalahan baku terhadap uji-t biasa
    if v <= 0:
        return float("nan"), float("nan"), float(faktor)
    t = d.mean() / np.sqrt((1.0 / J + n_uji / n_latih) * v)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1)), float(faktor)


def holm(p):
    """p terkoreksi Holm-Bonferroni (Holm, 1979) untuk satu keluarga pembanding."""
    p = np.asarray(p, float)
    m = len(p)
    urut = np.argsort(p)
    out = np.empty(m)
    maks = 0.0
    for i, k in enumerate(urut):
        maks = max(maks, (m - i) * p[k])
        out[k] = min(1.0, maks)
    return out


def fmt(x, nd=3):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}".replace(".", ",")
