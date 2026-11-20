"""B1 - regresi Ridge pada TF-IDF kata 1-2 gram, target K2.

Rujukan : Shabaruddin & Sim (2026), Bagian 4.6 (T2a): "Ridge on TF-IDF word 1-2 grams"; Bagian 4.7: hiperparameter dipilih dalam
          3 lipatan dalam yang hanya memakai data latih lipatan luar; Bagian 4.10: semua transformasi (termasuk kosakata TF-IDF dan IDF)
          ditaksir di dalam data latih saja; prediksi tidak dibulatkan dan tidak dipotong dalam validasi silang.
Pilihan kita (tidak disebut makalah): min_df = 2, kisi alpha, galat kuadrat rerata sebagai kriteria pemilihan alpha, pemisahan lipatan dalam.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import common as C

KODE = "B1"
NAMA = C.NAMA["B1"]
RUJUKAN = ["Shabaruddin & Sim (2026), Frontiers in Education 11, 1989989, Bagian 4.6 (T2a), 4.7, 4.10"]
PILIHAN_KITA = ["min_df = 2", "kisi alpha {0,1; 0,3; 1; 3; 10; 30}", "kriteria pemilihan alpha = galat kuadrat rerata", "seed lipatan dalam = seed * 1000 + lipatan"]


def _tfidf(cfg):
    return TfidfVectorizer(ngram_range=tuple(cfg["ridge_ngram"]), min_df=cfg["ridge_min_df"])


def pilih_alpha(teks, y, cfg, seed_dalam):
    """Pilih alpha dengan lipatan dalam; TF-IDF ditaksir ulang di tiap lipatan dalam (hanya pada bagian latih dalam)."""
    kf = KFold(cfg["ridge_lipatan_dalam"], shuffle=True, random_state=seed_dalam)
    alphas = list(cfg["ridge_alpha"])
    galat = np.zeros(len(alphas))
    for tr, va in kf.split(teks):
        vec = _tfidf(cfg)
        Xtr = vec.fit_transform([teks[i] for i in tr])
        Xva = vec.transform([teks[i] for i in va])
        for j, a in enumerate(alphas):
            m = Ridge(alpha=a).fit(Xtr, y[tr])
            galat[j] += np.sum((m.predict(Xva) - y[va]) ** 2)
    return alphas[int(np.argmin(galat))], (galat / len(teks)).tolist()


def jalankan(df, idx_latih, idx_uji, seed, lipatan, cfg, opsi=None):
    idx_latih, idx_uji = np.asarray(idx_latih), np.asarray(idx_uji)
    teks = [str(t).strip() for t in df.Text.values[idx_latih]]
    y = df.rerata.values[idx_latih].astype(float)
    alpha, mse_dalam = pilih_alpha(teks, y, cfg, seed * 1000 + lipatan)
    vec = _tfidf(cfg)
    X = vec.fit_transform(teks)
    model = Ridge(alpha=alpha).fit(X, y)
    pred = model.predict(vec.transform([str(t).strip() for t in df.Text.values[idx_uji]]))
    return dict(pred=pred, probs=None, idx_uji=idx_uji,
                log=dict(label="B1", alpha=alpha, mse_dalam=mse_dalam, n_fitur=int(X.shape[1])))
