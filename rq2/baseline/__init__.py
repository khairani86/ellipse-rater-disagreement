"""Satu berkas per pembanding (B0 sampai B3); semuanya dilatih/ditaksir pada target K2 (rerata dua penilai) dan dinilai
dengan fungsi evaluasi yang sama dengan K2-K6.

Antarmuka yang sama:
    KODE, NAMA, RUJUKAN (daftar), PILIHAN_KITA (daftar)
    jalankan(df, idx_latih, idx_uji, seed, lipatan, cfg, opsi) -> dict(pred (n_uji,), probs=None, log, idx_uji)
Skor acuan sendiri pembanding = rerata dua penilai (targetnya K2), jadi QWK 'sendiri' = QWK pada rerata.
"""
import importlib

_MODUL = {
    "B0": "b0_konstanta",
    "B1": "b1_ridge_tfidf",
    "B2": "b2_bert_standar",
    "B3": "b3_deberta_regresi",
}


def muat(kode):
    return importlib.import_module("baseline." + _MODUL[kode])
