"""K6 - skor harapan dari model many-facet Rasch pada skor Overall, hanya keketatan penilai.

Langkah (semuanya memakai HANYA esai latih lipatan itu untuk menaksir parameter):
  1. Model kategori-berdekatan (adjacent-category) dengan keketatan penilai s_r dan ambang bersama tau_k:
         log P(X = k) / P(X = k-1) = theta_i - s_r - tau_k ,   theta_i ~ N(mu, sigma^2)
     ditaksir dengan MMLE (kuadratur Gauss-Hermite, gradien analitik; mfrm_fast.py).
  2. Kemampuan esai theta_i diduga dengan EAP (rerata posterior) dari dua skornya.
  3. Target latih esai i = sebaran atas skor 1..5 yang dirata-ratakan atas semua penilai yang punya data latih:
         P_i[k] = (1/R) sum_r P(X = k | theta_hat_i, s_r, tau)
     Nilai harapannya  sum_k k P_i[k]  adalah skor K6 (Uto & Okano 2021, Persamaan 11, versi hanya-keketatan).
  4. Esai uji: theta_hat dari DUA skor esai uji itu sendiri dengan parameter dari lipatan latih (tanpa menaksir ulang);
     acuan_uji = nilai harapan sebaran yang sama.

Rujukan  : Uto & Okano (2021), IEEE TLT 14(6), 763-776, Persamaan 9-11 (dibaca penuh): theta dari model GMFRM dipakai sebagai skor esai,
           prediksi akhir = rerata atas semua penilai (1/R) sum_r sum_k k P(theta_hat, r, k).
           Wind dkk. (2018), IJT 18(1), 27-49: alasan K6 (mesin yang dilatih pada skor penilai meniru keketatannya); BUKAN cara membangun target.
Adaptasi (pilihan kita, diumumkan):
  - hanya keketatan penilai (tanpa konsistensi/diskriminasi dan tanpa ambang per penilai); 2 penilai per esai (Uto & Okano: 4);
  - MMLE dengan theta ~ N(mu, sigma^2) dan EAP, bukan Bayes penuh NUTS dengan prior N(0,1) seperti Uto & Okano;
    (JMLE seperti RQ1c tidak memberi theta berhingga pada esai berskor 5-5 atau 1-1);
  - target latih berupa sebaran lima skor untuk kepala 5 logit yang sama dengan K2-K6, bukan regresi MSE pada theta;
  - rata-rata atas penilai yang punya skor di lipatan latih (bukan semua penilai di data).
"""
import numpy as np
from scipy.special import logsumexp

import common as C
import mfrm_fast as F

KODE = "K6"
NAMA = C.NAMA["K6"]
RUJUKAN = ["Uto & Okano (2021), IEEE TLT 14(6), 763-776, Persamaan 9-11",
           "Wind dkk. (2018), IJT 18(1), 27-49 (alasan, bukan metode)"]
PILIHAN_KITA = ["hanya keketatan penilai, tanpa konsistensi dan ambang per penilai",
                "MMLE + EAP, bukan Bayes penuh (NUTS) dengan prior N(0,1)",
                "target latih = sebaran lima skor (kepala 5 logit), bukan regresi pada theta",
                "rata-rata atas penilai yang punya skor di lipatan latih"]


# ----------------------------------------------------------------- bagian model
def taksir(df, idx_latih, cfg):
    """Taksir model pada baris idx_latih saja. Kembalikan dict parameter + diagnostik."""
    raters, ia, ib, R = C.rater_arrays(df)
    idx_latih = np.asarray(idx_latih)
    xa, xb = df.x_a.values - 1, df.x_b.values - 1
    g = np.zeros(len(idx_latih), dtype=int)
    ia_, ib_, xa_, xb_, g_, cnt = F.compress(ia[idx_latih], ib[idx_latih], xa[idx_latih], xb[idx_latih], g)
    m = F.FastMFRM(ia_, ib_, xa_, xb_, g_, R, 1, Q=int(cfg["k6_node"]), cnt=cnt)
    res = m.fit(None, maxiter=int(cfg["k6_maxiter"]))
    tau, s, mu, sig = m.unpack(res.x)
    n_r = np.bincount(ia[idx_latih], minlength=R) + np.bincount(ib[idx_latih], minlength=R)
    return dict(tau=tau, s=s, mu=float(mu[0]), sig=float(sig), n_r=n_r, ia=ia, ib=ib, R=R,
                nodes=m.nodes, logw=m.logw,
                diag=dict(sukses=bool(res.success), iterasi=int(res.nit), nll=float(res.fun),
                          grad_maks=float(np.abs(res.jac).max()), pesan=str(res.message)))


def theta_eap(par, ia, ib, xa, xb):
    """EAP theta untuk esai dengan penilai ia, ib dan skor xa, xb (0..4), parameter `par` dari taksir()."""
    tau, s = par["tau"], par["s"]
    cum = np.concatenate([[0.0], np.cumsum(tau)])
    theta_q = par["mu"] + par["sig"] * par["nodes"]                      # (Q,)
    ll = np.zeros((len(ia), len(theta_q)))
    for ir, xr in ((ia, xa), (ib, xb)):
        eta = theta_q[None, :] - s[ir][:, None]                          # (n, Q)
        U = eta[:, :, None] * F.KK - cum                                 # (n, Q, 5)
        ll += eta * xr[:, None] - cum[xr][:, None] - logsumexp(U, axis=2)
    a = ll + par["logw"][None, :]
    post = np.exp(a - logsumexp(a, axis=1, keepdims=True))
    return post @ theta_q


def sebaran_rata_penilai(par, theta):
    """P_i[k] = rerata atas penilai ber-data-latih dari P(X = k | theta_i, s_r, tau). (n, 5)."""
    cum = np.concatenate([[0.0], np.cumsum(par["tau"])])
    s = par["s"][par["n_r"] > 0]
    U = F.KK * (theta[:, None, None] - s[None, :, None]) - cum          # (n, R', 5)
    U = U - logsumexp(U, axis=2, keepdims=True)
    return np.exp(U).mean(axis=1)


# ----------------------------------------------------------------- antarmuka
def bangun(df, idx_latih, idx_uji, cfg):
    idx_latih, idx_uji = np.asarray(idx_latih), np.asarray(idx_uji)
    par = taksir(df, idx_latih, cfg)
    xa, xb = df.x_a.values - 1, df.x_b.values - 1
    th_lat = theta_eap(par, par["ia"][idx_latih], par["ib"][idx_latih], xa[idx_latih], xb[idx_latih])
    P = sebaran_rata_penilai(par, th_lat)
    th_uji = theta_eap(par, par["ia"][idx_uji], par["ib"][idx_uji], xa[idx_uji], xb[idx_uji])
    acuan = sebaran_rata_penilai(par, th_uji) @ C.SKALA
    info = dict(n_latih=int(len(idx_latih)), n_penilai_dipakai=int((par["n_r"] > 0).sum()), mu=par["mu"], sigma=par["sig"],
                tau=[float(t) for t in par["tau"]], keketatan_min=float(par["s"][par["n_r"] > 0].min()),
                keketatan_maks=float(par["s"][par["n_r"] > 0].max()), **par["diag"])
    return dict(idx=idx_latih, P=P, acuan_uji=acuan, info=info, _par=par, _theta_latih=th_lat, _theta_uji=th_uji)
