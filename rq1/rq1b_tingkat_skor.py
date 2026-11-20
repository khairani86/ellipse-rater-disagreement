"""rq1b_tingkat_skor.py - RQ1b: pada skor berapa dua penilai paling sering berbeda.

Satu berkas, hanya butuh numpy dan pandas. Tidak memakai berkas lain di paket.

Dua cara melihat, mengikuti Kim (2026), Tabel 6 dan 7:
  B2 (utama)  bersyarat pada skor SATU penilai: kalau satu penilai memberi skor s, seberapa sering penilai lain memberi skor
              yang sama, seberapa sering selisihnya 2 poin atau lebih, dan berapa rata-rata skor penilai lain.
              Dihitung dari kedua arah (tiap penilai dalam pasangan sekali menjadi "satu penilai"), jadi tidak bergantung
              pada urutan slot. Satu baris esai menyumbang dua pengamatan.
  B1          menurut RERATA dua skor (1,0; 1,5; ... ; 5,0): persentase persis sama dan selisih >= 2.
              Beberapa sel ditentukan oleh skala, bukan temuan, dan ditandai:
                † persis sama tidak mungkin (rerata setengah-angka berarti selisih ganjil)
                § pasti persis sama (rerata 1,0 atau 5,0)
                ‡ selisih >= 2 tidak mungkin (rerata <= 1,5 atau >= 4,5)
Dihitung untuk skor keseluruhan (Overall), tiap aspek, dan gabungan enam aspek, lalu diulang pada tiga himpunan baris.
Ketidakpastian: CI 95% persentil dari bootstrap per pasangan penilai (B = 2000, seed 20261115).

Himpunan utama: baris ber-text_id_kaggle (himpunan resmi). Sensitivitas: semua baris yang dipakai (Overall > 0 pada kedua
penilai) dan baris tanpa text_id_kaggle. Himpunan utama bisa diganti dengan --utama semua.

Pakai:  python rq1b_tingkat_skor.py [csv] [folder_keluaran] [--utama resmi|semua]
        (bawaan: data/ellipse_raw.csv, hasil_rq1b/, himpunan utama resmi)
"""
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
GAB = "Gabungan 6 aspek"                            # enam aspek dijumlahkan (seperti Kim yang menggabungkan kriteria)
TINGKAT = np.arange(1, 5.01, 0.5)                   # rerata dua skor: 1,0 ... 5,0
NT = len(TINGKAT)                                   # 9 tingkat
_I = np.arange(K)
JAUH = (np.abs(np.subtract.outer(_I, _I)) >= 2).astype(float)   # sel tabel silang dengan selisih >= 2
UTAMA = "resmi"                                     # himpunan utama: "resmi" (baris ber-text_id_kaggle) atau "semua"
HIMPUNAN = {"resmi": "Baris ber-text_id_kaggle (himpunan resmi)", "semua": "Semua baris", "tanpa": "Baris tanpa text_id_kaggle"}


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
          f"esai unik: {df.Filename.nunique()}; pasangan penilai: {df.pair.nunique()}")
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
    print(f"Himpunan utama ({HIMPUNAN['resmi']}): {len(d)} dari {len(df)} baris; "
          f"esai unik: {d.Filename.nunique()}; pasangan penilai: {d.pair.nunique()}")
    return d


# ----------------------------------------------------------------------------------------------------------
# 2. Statistik per baris dan tabel dari jumlahnya
# ----------------------------------------------------------------------------------------------------------
def statistik_baris(a, b, ada):
    """Tiap baris -> 52 angka: 25 penanda sel tabel silang (skor a, skor b), lalu untuk tiap dari 9 tingkat rerata:
    penanda tingkat, penanda persis sama, penanda selisih >= 2. Baris tanpa skor valid diisi 0."""
    S = np.zeros((len(a), 25 + 3 * NT))
    i = np.where(ada)[0]
    S[i, (a[i] - 1) * K + (b[i] - 1)] = 1.0
    t = a[i] + b[i] - 2                             # 0..8  <->  rerata 1,0 ... 5,0
    d = np.abs(a[i] - b[i])
    S[i, 25 + t] = 1.0
    S[i, 25 + NT + t] = (d == 0)
    S[i, 25 + 2 * NT + t] = (d >= 2)
    return S


def tabel_bersyarat(T):
    """T = jumlah statistik (..., 52). Bersyarat pada skor satu penilai, dari kedua arah.
    sym = O + O^T: sel (s, j) = banyak pengamatan 'satu penilai memberi s, penilai lain memberi j'."""
    O = T[..., :25].reshape(T.shape[:-1] + (K, K))
    sym = O + np.swapaxes(O, -1, -2)
    n = sym.sum(-1)
    sama = np.diagonal(sym, axis1=-2, axis2=-1)
    jauh = (sym * JAUH).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(n=n, n_sama=sama, n_ge2=jauh, same=100 * sama / n, ge2=100 * jauh / n,
                    mean_other=(sym * (_I + 1)[None, :]).sum(-1) / n)


def tabel_rerata(T):
    """T = jumlah statistik (..., 52). Menurut rerata dua skor (9 tingkat)."""
    L = T[..., 25:]
    n = L[..., :NT]
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(n=n, n_exact=L[..., NT:2 * NT], n_ge2=L[..., 2 * NT:],
                    exact=100 * L[..., NT:2 * NT] / n, ge2=100 * L[..., 2 * NT:] / n)
# ----------------------------------------------------------------------------------------------------------
# 3. Bootstrap per pasangan penilai dan tabel hasil
# ----------------------------------------------------------------------------------------------------------
def hitung(d, B=B, seed=SEED):
    """Tabel titik dan B ulangan bootstrap (pasangan penilai diambil ulang dengan pengembalian) untuk tiap
    kriteria dan gabungan enam aspek. Satu undian pasangan dipakai bersama oleh semuanya."""
    kl, _ = pd.factorize(d.pair)
    nk = kl.max() + 1
    M = {}
    for c in KRITERIA:
        a, b = d[f"{c}_1"].values.astype(int), d[f"{c}_2"].values.astype(int)
        S = statistik_baris(a, b, (a > 0) & (b > 0))
        M[c] = np.stack([np.bincount(kl, weights=S[:, j], minlength=nk) for j in range(S.shape[1])], axis=1)
    M[GAB] = sum(M[c] for c in ASPEK)
    undian = np.random.default_rng(seed).integers(0, nk, (B, nk))
    cacah = np.stack([np.bincount(u, minlength=nk) for u in undian]).astype(float)
    rr, rb = [], []
    for c in M:
        tp_r, bt_r = tabel_rerata(M[c].sum(0)), tabel_rerata(cacah @ M[c])
        for i, m in enumerate(TINGKAT):
            if tp_r["n"][i] == 0:
                continue
            dmaks = int(2 * min(m - 1, 5 - m))      # selisih terbesar yang mungkin pada rerata m
            r = dict(kriteria=c, rerata=m, n=int(tp_r["n"][i]), n_exact=int(tp_r["n_exact"][i]),
                     n_ge2=int(tp_r["n_ge2"][i]), persis_tak_mungkin=bool(m % 1 != 0),
                     persis_pasti=bool(m in (1.0, 5.0)), selisih2_tak_mungkin=bool(dmaks < 2))
            for k in ("exact", "ge2"):
                lo, hi = np.nanpercentile(bt_r[k][:, i], [2.5, 97.5])
                r[k], r[k + "_lo"], r[k + "_hi"] = float(tp_r[k][i]), float(lo), float(hi)
            rr.append(r)
        tp_b, bt_b = tabel_bersyarat(M[c].sum(0)), tabel_bersyarat(cacah @ M[c])
        for s in range(K):
            r = dict(kriteria=c, skor_satu_penilai=s + 1, n_pengamatan=int(tp_b["n"][s]),
                     n_sama=int(tp_b["n_sama"][s]), n_ge2=int(tp_b["n_ge2"][s]))
            for k in ("same", "ge2", "mean_other"):
                lo, hi = np.nanpercentile(bt_b[k][:, s], [2.5, 97.5])
                r[k], r[k + "_lo"], r[k + "_hi"] = float(tp_b[k][s]), float(lo), float(hi)
            rb.append(r)
    return pd.DataFrame(rr), pd.DataFrame(rb), dict(kl=kl, nk=nk, cacah=cacah, M=M)


def sensitivitas(df):
    """Ulangi pada tiga himpunan baris (himpunan utama lebih dulu) dari semua baris; lewati bila text_id_kaggle tidak ada."""
    if "text_id_kaggle" not in df.columns:
        return None
    hasil_r, hasil_b = [], []
    for kunci in [UTAMA] + [k for k in HIMPUNAN if k != UTAMA]:
        nama = HIMPUNAN[kunci] + (" (utama)" if kunci == UTAMA else "")
        d = df[mask_himpunan(df, kunci)].reset_index(drop=True)
        if len(d) == 0 or d.pair.nunique() < 2:
            continue
        tr, tb, _ = hitung(d)
        tr.insert(0, "himpunan", nama)
        tb.insert(0, "himpunan", nama)
        hasil_r.append(tr)
        hasil_b.append(tb)
    if len(hasil_b) < 2:
        return None
    return pd.concat(hasil_r, ignore_index=True), pd.concat(hasil_b, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------
# 4. Pemeriksaan: hitung ulang dengan cara lain dan bandingkan
# ----------------------------------------------------------------------------------------------------------
def bersyarat_langsung(x, y):
    """Tabel bersyarat dari dua kolom skor: gabungkan kedua arah, lalu hitung per skor satu penilai."""
    a, b = np.concatenate([x, y]), np.concatenate([y, x])
    hasil = []
    for s in range(1, K + 1):
        m = a == s
        hasil.append((m.sum(), 100 * (b[m] == s).mean(), 100 * (np.abs(b[m] - s) >= 2).mean(), b[m].mean()))
    return hasil


def rerata_langsung(x, y):
    """Tabel menurut rerata dari dua kolom skor, dengan pengelompokan biasa (groupby)."""
    t = pd.DataFrame({"r": (x + y) / 2, "sama": x == y, "jauh": np.abs(x - y) >= 2}).groupby("r")
    return t.size(), 100 * t.sama.mean(), 100 * t.jauh.mean()


def periksa(d, titik_r, titik_b, info):
    """Hitung angka utama dengan cara lain dan bandingkan; hentikan program bila ada yang tidak cocok."""
    x, y = d.Overall_1.values, d.Overall_2.values
    cek = []
    tb = titik_b[titik_b.kriteria == "Overall"].reset_index(drop=True)
    for s, (n, sama, ge2, rata) in enumerate(bersyarat_langsung(x, y)):
        cek += [(f"Bersyarat skor {s + 1}: n", tb.n_pengamatan[s], n), (f"Bersyarat skor {s + 1}: sama %", tb.same[s], sama),
                (f"Bersyarat skor {s + 1}: selisih >= 2 %", tb.ge2[s], ge2),
                (f"Bersyarat skor {s + 1}: rerata penilai lain", tb.mean_other[s], rata)]
    tr = titik_r[titik_r.kriteria == "Overall"].set_index("rerata")
    n_l, ex_l, g2_l = rerata_langsung(x, y)
    for m in n_l.index:
        cek += [(f"Rerata {m}: n", tr.n[m], n_l[m]), (f"Rerata {m}: persis %", tr.exact[m], ex_l[m]),
                (f"Rerata {m}: selisih >= 2 %", tr.ge2[m], g2_l[m])]
    cek.append(("Jumlah pengamatan bersyarat = 2 x baris", tb.n_pengamatan.sum(), 2 * len(d)))
    cek.append(("Jumlah baris menurut rerata = baris", tr.n.sum(), len(d)))
    # satu ulangan bootstrap dihitung dua cara: vektor (cacah @ M) dan mengambil baris pasangan satu per satu
    kl, cacah = info["kl"], info["cacah"]
    baris = np.concatenate([np.where(kl == k)[0] for k in range(info["nk"]) for _ in range(int(cacah[0, k]))])
    vek = tabel_bersyarat(cacah[0] @ info["M"]["Overall"])
    lang = bersyarat_langsung(x[baris], y[baris])
    for s in range(K):
        cek.append((f"Ulangan bootstrap, skor {s + 1}: selisih >= 2 % (vektor vs ambil-baris)", vek["ge2"][s], lang[s][2]))
    # sel yang ditentukan skala harus berperilaku sesuai penandanya
    for _, r in tr.reset_index().iterrows():
        if r.persis_tak_mungkin:
            cek.append((f"Sel † rerata {r.rerata}: persis % = 0", r.exact, 0.0))
        if r.persis_pasti:
            cek.append((f"Sel § rerata {r.rerata}: persis % = 100", r.exact, 100.0))
        if r.selisih2_tak_mungkin:
            cek.append((f"Sel ‡ rerata {r.rerata}: selisih >= 2 % = 0", r.ge2, 0.0))
    ok = True
    for nama, a, b in cek:
        ok &= abs(float(a) - float(b)) < 1e-9
    print(f"  {'LOLOS' if ok else 'GAGAL'}  {len(cek)} pemeriksaan (tabel bersyarat, tabel menurut rerata, jumlah, "
          f"bootstrap vektor vs ambil-baris, penanda sel)")
    if not ok:
        for nama, a, b in cek:
            if abs(float(a) - float(b)) >= 1e-9:
                print(f"  GAGAL  {nama}: {a} vs {b}")
        raise SystemExit("Pemeriksaan gagal.")
# ----------------------------------------------------------------------------------------------------------
# 5. Keluaran
# ----------------------------------------------------------------------------------------------------------
def f(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def fi(n):
    return f"{int(n):,}".replace(",", ".")


def fci(r, k, nd=1):
    return f"{f(r[k], nd)} [{f(r[k + '_lo'], nd)}; {f(r[k + '_hi'], nd)}]"


def md(kepala, baris):
    return "\n".join(["| " + " | ".join(kepala) + " |", "|" + "---|" * len(kepala)] +
                     ["| " + " | ".join(r) + " |" for r in baris]) + "\n"


def ringkasan(tr, tb, sens):
    s = ["# RQ1b: pada skor berapa dua penilai berbeda\n",
         f"Himpunan utama: {HIMPUNAN[UTAMA]}.\n",
         "CI 95% dari bootstrap per pasangan penilai. Sel † tidak bisa persis sama (rerata setengah-angka), sel § pasti persis "
         "sama (rerata 1,0 atau 5,0), sel ‡ tidak bisa berselisih ≥ 2 (rerata ≤ 1,5 atau ≥ 4,5): konsekuensi skala, bukan temuan.\n"]
    for c in ["Overall", GAB]:
        s.append(f"## B2. Bersyarat pada skor satu penilai: {c}\n")
        s.append(md(["Skor satu penilai", "Pengamatan", "Penilai lain sama % [CI]", "Selisih ≥ 2 % [CI]", "Rerata penilai lain [CI]"],
                    [[str(int(r.skor_satu_penilai)), fi(r.n_pengamatan), fci(r, "same"), fci(r, "ge2"), fci(r, "mean_other", 2)]
                     for _, r in tb[tb.kriteria == c].iterrows()]))
    for k, judul in (("ge2", "Selisih ≥ 2 % menurut skor satu penilai, tiap kriteria [CI]"),
                     ("same", "Penilai lain memberi skor sama % menurut skor satu penilai, tiap kriteria [CI]")):
        s.append(f"## {judul}\n")
        s.append(md(["Kriteria"] + [f"Skor {j}" for j in range(1, K + 1)],
                    [[c] + [fci(r, k) for _, r in tb[tb.kriteria == c].iterrows()] for c in KRITERIA + [GAB]]))
    for c in ["Overall", GAB]:
        s.append(f"## B1. Menurut rerata dua skor: {c}\n")
        baris = []
        for _, r in tr[tr.kriteria == c].iterrows():
            ex = "†" if r.persis_tak_mungkin else ("§" if r.persis_pasti else "")
            g2 = "‡" if r.selisih2_tak_mungkin else ""
            baris.append([f(r.rerata), fi(r.n), fci(r, "exact") + ex, fci(r, "ge2") + g2])
        s.append(md(["Rerata", "Baris", "Persis % [CI]", "Selisih ≥ 2 % [CI]"], baris))
    if sens is not None:
        sr, sb = sens
        s.append("## Sensitivitas menurut himpunan baris: bersyarat pada skor satu penilai, skor keseluruhan\n")
        baris = []
        for h, g in sb[sb.kriteria == "Overall"].groupby("himpunan", sort=False):
            baris.append([h] + [f"{fi(r.n_pengamatan)}: {f(r.same)}% sama, {f(r.ge2)}% ≥ 2" for _, r in g.iterrows()])
        s.append(md(["Himpunan"] + [f"Skor {j}" for j in range(1, K + 1)], baris))
    return "\n".join(s)


def main(csv=None, keluar=None):
    csv = csv or os.path.join(HERE, "data", "ellipse_raw.csv")
    keluar = keluar or os.path.join(HERE, "hasil_rq1b")
    os.makedirs(keluar, exist_ok=True)
    df_semua = muat(csv)
    df = pilih_utama(df_semua)
    tr, tb, info = hitung(df)
    print("Pemeriksaan hitungan:")
    periksa(df, tr, tb, info)
    tr.to_csv(os.path.join(keluar, "rq1b_menurut_rerata.csv"), index=False)
    tb.to_csv(os.path.join(keluar, "rq1b_bersyarat_skor_satu_penilai.csv"), index=False)
    sens = sensitivitas(df_semua)
    if sens is not None:
        sens[0].to_csv(os.path.join(keluar, "rq1b_himpunan_bagian_menurut_rerata.csv"), index=False)
        sens[1].to_csv(os.path.join(keluar, "rq1b_himpunan_bagian_bersyarat.csv"), index=False)
    teks = ringkasan(tr, tb, sens)
    with open(os.path.join(keluar, "ringkasan_rq1b.md"), "w", encoding="utf-8") as fh:
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
