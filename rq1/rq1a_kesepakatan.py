"""rq1a_kesepakatan.py - RQ1a: seberapa setuju dua penilai, dan seberapa jauh dari batas yang dapat dicapai.

Satu berkas, hanya butuh numpy dan pandas. Tidak memakai berkas lain di paket.

Yang dihitung, untuk skor keseluruhan (Overall) dan enam aspek:
  QWK        kappa berbobot kuadrat, penilai slot 1 vs slot 2 (ukuran utama)
  QWK_sim    QWK versi simetris (tidak bergantung pada urutan slot)
  Pearson    korelasi (hanya asosiasi; tidak melihat beda keketatan)
  ICC1       ICC(1,1) satu arah: bagian penyebaran skor yang berasal dari perbedaan esai
  kappa_HL   batas model yang galatnya sebesar satu penilai   = sqrt(rho_1 * rho_Y)   (Uto, 2026)
  kappa_max  batas model sempurna                              = sqrt(rho_Y)           (Uto, 2026)
             dengan rho_1 = ICC1 dan rho_Y = 2 * rho_1 / (1 + rho_1)
  Exact, Within1, Ge2   persentase baris dengan selisih 0, paling banyak 1, dan 2 poin atau lebih
Ketidakpastian: CI 95% persentil dari bootstrap per pasangan penilai (B = 2000, seed 20261115).
Juga: tabel silang 5x5, selisih QWK tiap aspek terhadap Overall dan antar aspek, jarak ke batas, dan sensitivitas pada
tiga himpunan baris.

Himpunan utama: baris ber-text_id_kaggle (himpunan resmi). Sensitivitas: semua baris yang dipakai (Overall > 0 pada kedua
penilai) dan baris tanpa text_id_kaggle. Himpunan utama bisa diganti dengan --utama semua.

Pakai:  python rq1a_kesepakatan.py [csv] [folder_keluaran] [--utama resmi|semua]
        (bawaan: data/ellipse_raw.csv, hasil_rq1a/, himpunan utama resmi)
"""
import itertools
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
B = 2000                                            # banyak ulangan bootstrap
SEED = 20261115
K = 5                                               # skor 1..5
ASPEK = ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]
KRITERIA = ["Overall"] + ASPEK
UKURAN = ["QWK", "QWK_sim", "Pearson", "ICC1", "kappa_HL", "kappa_max", "Exact", "Within1", "Ge2"]
UTAMA = "resmi"                                     # himpunan utama: "resmi" (baris ber-text_id_kaggle) atau "semua"
HIMPUNAN = {"resmi": "Baris ber-text_id_kaggle (himpunan resmi)", "semua": "Semua baris", "tanpa": "Baris tanpa text_id_kaggle"}

_I = np.arange(K)
BOBOT = np.subtract.outer(_I, _I) ** 2 / (K - 1) ** 2        # bobot kuadrat: 0 ... 1
DEKAT = (np.abs(np.subtract.outer(_I, _I)) <= 1).astype(float)  # sel dengan selisih <= 1
JAUH = (np.abs(np.subtract.outer(_I, _I)) >= 2).astype(float)   # sel dengan selisih >= 2


# ----------------------------------------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------------------------------------
def muat(csv):
    """Baca berkas mentah. Baris dipakai bila skor Overall kedua penilai > 0 (0 = tidak ada skor)."""
    if not os.path.exists(csv):
        raise SystemExit(f"Berkas tidak ditemukan: {csv}\nTaruh berkas ELLIPSE mentah di data/ellipse_raw.csv")
    df = pd.read_csv(csv)
    perlu = ["Filename", "Rater_1", "Rater_2"] + [f"{c}_{s}" for c in KRITERIA for s in (1, 2)]
    hilang = [c for c in perlu if c not in df.columns]
    if hilang:
        raise SystemExit(f"Kolom tidak ada di berkas: {hilang}")
    n_awal = len(df)
    df = df[(df.Overall_1 > 0) & (df.Overall_2 > 0)].reset_index(drop=True)

    def no(s):                                      # "rater_9" -> 9, supaya pasangan selalu urut sama
        try:
            return int(str(s).split("_")[-1])
        except ValueError:
            return 10 ** 9

    df["pair"] = [" - ".join(sorted([a, b], key=no)) for a, b in zip(df.Rater_1, df.Rater_2)]
    print(f"Baris di berkas: {n_awal}; dipakai (Overall > 0 pada kedua penilai): {len(df)}; "
          f"esai unik: {df.Filename.nunique()}; penilai: {len(set(df.Rater_1) | set(df.Rater_2))}; "
          f"pasangan penilai: {df.pair.nunique()}")
    return df, n_awal


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
    print(f"Himpunan utama ({HIMPUNAN['resmi']}): {len(d)} dari {len(df)} baris; "
          f"esai unik: {d.Filename.nunique()}; pasangan penilai: {d.pair.nunique()}")
    return d


# ----------------------------------------------------------------------------------------------------------
# 2. Ukuran kesepakatan, dihitung dari statistik ringkas supaya bootstrap cepat
# ----------------------------------------------------------------------------------------------------------
def statistik_baris(a, b, ada):
    """Tiap baris -> 30 angka: 25 kolom penanda sel (skor a, skor b) di tabel silang, lalu a, b, a^2, b^2, a*b.
    Baris tanpa skor valid diisi 0. Jumlah kolom-kolom ini cukup untuk semua ukuran di bawah."""
    S = np.zeros((len(a), 30))
    i = np.where(ada)[0]
    S[i, (a[i] - 1) * K + (b[i] - 1)] = 1.0
    S[:, 25], S[:, 26] = np.where(ada, a, 0), np.where(ada, b, 0)
    S[:, 27], S[:, 28] = np.where(ada, a * a, 0), np.where(ada, b * b, 0)
    S[:, 29] = np.where(ada, a * b, 0)
    return S


def qwk(O):
    """QWK dari tabel silang O (baris = skor slot 1, kolom = skor slot 2):
    1 - (selisih kuadrat teramati) / (selisih kuadrat bila skor kedua penilai saling bebas)."""
    O = np.asarray(O, float)
    n = O.sum((-1, -2))
    harap = O.sum(-1)[..., :, None] * O.sum(-2)[..., None, :] / n[..., None, None]
    return 1.0 - (BOBOT * O).sum((-1, -2)) / (BOBOT * harap).sum((-1, -2))


def ukuran(T):
    """T = jumlah statistik_baris (bentuk (..., 30)); kembalikan semua ukuran sebagai dict."""
    T = np.asarray(T, float)
    O = T[..., :25].reshape(T.shape[:-1] + (K, K))
    n = O.sum((-1, -2))
    Sa, Sb, Saa, Sbb, Sab = (T[..., j] for j in range(25, 30))
    o = {"n": n, "O": O}
    o["QWK"] = qwk(O)
    o["QWK_sim"] = qwk((O + np.swapaxes(O, -1, -2)) / 2)
    o["Pearson"] = (n * Sab - Sa * Sb) / np.sqrt((n * Saa - Sa ** 2) * (n * Sbb - Sb ** 2))
    # ICC(1,1): esai = kelompok, dua penilai tiap esai.
    #   MSB = ragam antar esai, MSW = ragam di dalam esai (dua skor berbeda untuk esai yang sama)
    rata = (Sa + Sb) / 2                            # jumlah rerata-esai
    kuad = (Saa + 2 * Sab + Sbb) / 4                # jumlah kuadrat rerata-esai
    msb = 2 * (kuad - rata ** 2 / n) / (n - 1)
    msw = (Saa - 2 * Sab + Sbb) / 2 / n
    rho1 = (msb - msw) / (msb + msw)                # ICC satu penilai
    rhoY = (msb - msw) / msb                        # keandalan rerata dua penilai = 2*rho1/(1+rho1)
    o["ICC1"], o["rho_Y"] = rho1, rhoY
    with np.errstate(invalid="ignore"):
        o["kappa_max"] = np.sqrt(np.clip(rhoY, 0, None))
        o["kappa_HL"] = np.sqrt(np.clip(rho1 * rhoY, 0, None))
    o["Exact"] = 100 * np.trace(O, axis1=-2, axis2=-1) / n
    o["Within1"] = 100 * (O * DEKAT).sum((-1, -2)) / n
    o["Ge2"] = 100 * (O * JAUH).sum((-1, -2)) / n
    o["MeanDiff"] = (Sa - Sb) / n                   # rerata skor slot 1 dikurangi slot 2
    return o


# ----------------------------------------------------------------------------------------------------------
# 3. Bootstrap per pasangan penilai
# ----------------------------------------------------------------------------------------------------------
def hitung(d, B=B, seed=SEED):
    """Ukuran titik dan B ulangan bootstrap (pasangan penilai diambil ulang dengan pengembalian).
    Satu undian pasangan dipakai bersama oleh semua kriteria, jadi selisih antar kriteria tetap berpasangan."""
    kl, _ = pd.factorize(d.pair)
    nk = kl.max() + 1
    titik, M = {}, {}
    for c in KRITERIA:
        a, b = d[f"{c}_1"].values.astype(int), d[f"{c}_2"].values.astype(int)
        S = statistik_baris(a, b, (a > 0) & (b > 0))
        titik[c] = ukuran(S.sum(0))
        M[c] = np.stack([np.bincount(kl, weights=S[:, j], minlength=nk) for j in range(30)], axis=1)  # per pasangan
    undian = np.random.default_rng(seed).integers(0, nk, (B, nk))
    cacah = np.stack([np.bincount(u, minlength=nk) for u in undian]).astype(float)   # berapa kali tiap pasangan terambil
    boot = {c: ukuran(cacah @ M[c]) for c in KRITERIA}
    return titik, boot, dict(kl=kl, nk=nk, cacah=cacah, M=M)


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def tabel_utama(titik, boot):
    """Satu baris per kriteria: nilai titik + CI untuk tiap ukuran, jarak ke batas, selisih terhadap Overall."""
    baris = []
    for c in KRITERIA:
        r = {"kriteria": c, "n": int(titik[c]["n"]), "MeanDiff": float(titik[c]["MeanDiff"])}
        for m in UKURAN:
            lo, hi = ci(boot[c][m])
            r.update({m: float(titik[c][m]), m + "_lo": lo, m + "_hi": hi})
        for nama, kap in (("jarak_HL", "kappa_HL"), ("jarak_max", "kappa_max")):
            r[nama] = float(titik[c][kap] - titik[c]["QWK"])
            r[nama + "_lo"], r[nama + "_hi"] = ci(boot[c][kap] - boot[c]["QWK"])
        if c != "Overall":
            r["selisih_QWK_vs_Overall"] = float(titik[c]["QWK"] - titik["Overall"]["QWK"])
            r["selisih_lo"], r["selisih_hi"] = ci(boot[c]["QWK"] - boot["Overall"]["QWK"])
        baris.append(r)
    return pd.DataFrame(baris)


def selisih_antar_aspek(titik, boot):
    """Selisih QWK untuk tiap pasangan aspek (15 pasang). CI dari undian bootstrap yang sama (berpasangan).
    Tidak ada koreksi uji berganda: selisih yang batas CI-nya dekat 0 tidak boleh dibaca sebagai pasti."""
    baris = []
    for a, b in itertools.combinations(ASPEK, 2):
        lo, hi = ci(boot[a]["QWK"] - boot[b]["QWK"])
        baris.append(dict(aspek_a=a, aspek_b=b, selisih=float(titik[a]["QWK"] - titik[b]["QWK"]), lo=lo, hi=hi,
                          tidak_memuat_nol=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(baris).sort_values("selisih").reset_index(drop=True)


def tabel_silang(titik):
    """Tabel silang 5x5 (jumlah baris) untuk tiap kriteria, bentuk panjang."""
    baris = []
    for c in KRITERIA:
        O = titik[c]["O"]
        for i in range(K):
            for j in range(K):
                baris.append(dict(kriteria=c, skor_slot1=i + 1, skor_slot2=j + 1, n=int(O[i, j])))
    return pd.DataFrame(baris)


def sensitivitas(df):
    """Hitung ulang untuk tiga himpunan baris (himpunan utama lebih dulu) dari semua baris; lewati bila text_id_kaggle tidak ada."""
    if "text_id_kaggle" not in df.columns:
        return None
    hasil = []
    for kunci in [UTAMA] + [k for k in HIMPUNAN if k != UTAMA]:
        nama = HIMPUNAN[kunci] + (" (utama)" if kunci == UTAMA else "")
        d = df[mask_himpunan(df, kunci)].reset_index(drop=True)
        if len(d) == 0 or d.pair.nunique() < 2:
            continue
        titik, boot, info = hitung(d)
        t = tabel_utama(titik, boot)
        t.insert(0, "himpunan", nama)
        t.insert(2, "n_pasangan", info["nk"])
        hasil.append(t)
    return pd.concat(hasil, ignore_index=True) if len(hasil) > 1 else None


# ----------------------------------------------------------------------------------------------------------
# 4. Pemeriksaan: hitung ulang dengan cara lain dan bandingkan
# ----------------------------------------------------------------------------------------------------------
def qwk_langsung(x, y):
    """QWK dari dua kolom skor, tanpa statistik ringkas."""
    O = np.zeros((K, K))
    for i, j in zip(x, y):
        O[i - 1, j - 1] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    w = np.array([[(i - j) ** 2 / (K - 1) ** 2 for j in range(K)] for i in range(K)])
    return 1 - (w * O).sum() / (w * E).sum()


def icc_anova(x, y):
    """ICC(1,1) dari analisis ragam satu arah yang ditulis lengkap (esai = kelompok, dua penilai tiap esai)."""
    X = np.column_stack([x, y]).astype(float)
    n = len(X)
    rata_esai = X.mean(1)
    msb = 2 * ((rata_esai - X.mean()) ** 2).sum() / (n - 1)
    msw = ((X - rata_esai[:, None]) ** 2).sum() / n
    return (msb - msw) / (msb + msw)


def periksa(d, titik, info):
    """Hitung angka utama dengan cara lain dan bandingkan; hentikan program bila ada yang tidak cocok."""
    x, y = d.Overall_1.values, d.Overall_2.values
    cek = [("QWK Overall: tabel silang vs hitung langsung", titik["Overall"]["QWK"], qwk_langsung(x, y)),
           ("ICC1 Overall: statistik ringkas vs analisis ragam", titik["Overall"]["ICC1"], icc_anova(x, y)),
           ("Pearson Overall: statistik ringkas vs np.corrcoef", titik["Overall"]["Pearson"], np.corrcoef(x, y)[0, 1])]
    try:                                            # pemeriksaan tambahan bila scikit-learn terpasang
        from sklearn.metrics import cohen_kappa_score
        cek.append(("QWK Overall: vs scikit-learn", titik["Overall"]["QWK"],
                    cohen_kappa_score(x, y, weights="quadratic")))
    except ImportError:
        pass
    # satu ulangan bootstrap dihitung dua cara: vektor (cacah @ M) dan mengambil baris pasangan satu per satu
    kl, cacah = info["kl"], info["cacah"]
    u = 0
    baris = np.concatenate([np.where(kl == k)[0] for k in range(info["nk"]) for _ in range(int(cacah[u, k]))])
    cek.append(("QWK Overall: ulangan bootstrap vektor vs ambil-baris", qwk(
        (info["cacah"][u] @ info["M"]["Overall"])[:25].reshape(K, K)),
        qwk_langsung(x[baris], y[baris])))
    ok = True
    for nama, a, b in cek:
        lolos = abs(a - b) < 1e-9
        ok &= lolos
        print(f"  {'LOLOS' if lolos else 'GAGAL'}  {nama}: {a:.10f} vs {b:.10f}")
    if not ok:
        raise SystemExit("Pemeriksaan gagal.")


# ----------------------------------------------------------------------------------------------------------
# 5. Keluaran
# ----------------------------------------------------------------------------------------------------------
def f(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def fci(r, m, nd=3):
    return f"{f(r[m], nd)} [{f(r[m + '_lo'], nd)}; {f(r[m + '_hi'], nd)}]"


def md(kepala, baris):
    return "\n".join(["| " + " | ".join(kepala) + " |", "|" + "---|" * len(kepala)] +
                     ["| " + " | ".join(r) + " |" for r in baris]) + "\n"


def ringkasan(t, sens, titik, antar):
    s = ["# RQ1a: seberapa setuju dua penilai\n",
         f"Himpunan utama: {HIMPUNAN[UTAMA]}.\n",
         "## Kesepakatan dan batas (CI 95%, bootstrap per pasangan penilai)\n",
         md(["Kriteria", "n", "QWK", "Pearson", "ICC(1,1)", "Persis sama %", "Selisih ≥ 2 %", "κ_HL", "κ_max"],
            [[r.kriteria, str(r.n), fci(r, "QWK"), f(r.Pearson), f(r.ICC1), f(r.Exact, 1), f(r.Ge2, 1),
              fci(r, "kappa_HL"), fci(r, "kappa_max")] for _, r in t.iterrows()]),
         "## Aspek dibanding skor keseluruhan dan jarak ke batas\n",
         md(["Aspek", "QWK aspek − QWK Overall [CI]", "κ_HL − QWK [CI]", "κ_max − QWK [CI]"],
            [[r.kriteria, f"{f(r.selisih_QWK_vs_Overall)} [{f(r.selisih_lo)}; {f(r.selisih_hi)}]",
              f"{f(r.jarak_HL)} [{f(r.jarak_HL_lo)}; {f(r.jarak_HL_hi)}]",
              f"{f(r.jarak_max)} [{f(r.jarak_max_lo)}; {f(r.jarak_max_hi)}]"] for _, r in t.iloc[1:].iterrows()]),
         "## Selisih QWK antar aspek (CI berpasangan, tanpa koreksi uji berganda)\n",
         md(["Aspek A", "Aspek B", "QWK A − QWK B [CI]", "CI tidak memuat 0"],
            [[r.aspek_a, r.aspek_b, f"{f(r.selisih)} [{f(r.lo)}; {f(r.hi)}]", "ya" if r.tidak_memuat_nol else "tidak"]
             for _, r in antar.iterrows()]),
         "## Tabel silang skor keseluruhan (baris = penilai slot 1, kolom = penilai slot 2)\n"]
    O = titik["Overall"]["O"].astype(int)
    s.append(md(["Slot 1 \\ Slot 2"] + [str(k) for k in range(1, K + 1)] + ["Jumlah"],
                [[str(i + 1)] + [str(v) for v in O[i]] + [str(O[i].sum())] for i in range(K)] +
                [["Jumlah"] + [str(v) for v in O.sum(0)] + [str(O.sum())]]))
    if sens is not None:
        s.append("## Sensitivitas menurut himpunan baris (QWK [CI 95%])\n")
        kolom = ["Himpunan", "n", "Overall"] + ASPEK + ["κ_HL Overall", "κ_max Overall", "Persis %", "≥ 2 %"]
        baris = []
        for h, g in sens.groupby("himpunan", sort=False):
            r0 = g[g.kriteria == "Overall"].iloc[0]
            baris.append([h, str(r0.n), fci(r0, "QWK")] +
                         [fci(g[g.kriteria == a].iloc[0], "QWK", 2) for a in ASPEK] +
                         [f(r0.kappa_HL), f(r0.kappa_max), f(r0.Exact, 1), f(r0.Ge2, 1)])
        s.append(md(kolom, baris))
    return "\n".join(s)


def main(csv=None, keluar=None):
    csv = csv or os.path.join(HERE, "data", "ellipse_raw.csv")
    keluar = keluar or os.path.join(HERE, "hasil_rq1a")
    os.makedirs(keluar, exist_ok=True)
    df_semua, _ = muat(csv)
    df = pilih_utama(df_semua)
    titik, boot, info = hitung(df)
    print("Pemeriksaan hitungan:")
    periksa(df, titik, info)
    t = tabel_utama(titik, boot)
    t.to_csv(os.path.join(keluar, "rq1a_per_kriteria.csv"), index=False)
    antar = selisih_antar_aspek(titik, boot)
    antar.to_csv(os.path.join(keluar, "rq1a_selisih_antar_aspek.csv"), index=False)
    tabel_silang(titik).to_csv(os.path.join(keluar, "rq1a_tabel_silang.csv"), index=False)
    sens = sensitivitas(df_semua)
    if sens is not None:
        sens.to_csv(os.path.join(keluar, "rq1a_himpunan_bagian.csv"), index=False)
    teks = ringkasan(t, sens, titik, antar)
    with open(os.path.join(keluar, "ringkasan_rq1a.md"), "w", encoding="utf-8") as fh:
        fh.write(teks)
    print("\n" + teks)
    print(f"Selesai. Berkas ada di: {keluar}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--utama" in argv:
        i = argv.index("--utama")
        UTAMA = argv[i + 1] if i + 1 < len(argv) else ""
        del argv[i:i + 2]
        if UTAMA not in ("resmi", "semua"):
            raise SystemExit("--utama harus resmi atau semua")
    main(*argv[:2])
