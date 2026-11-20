"""Satu berkas per metode skor acuan (K2 sampai K6). Tiap berkas dapat diuji sendiri (uji_kontrol/uji_target.py).

Antarmuka yang sama untuk semua berkas:
    KODE, NAMA, RUJUKAN (daftar), PILIHAN_KITA (daftar)
    bangun(df, idx_latih, idx_uji, cfg) -> dict(
        idx       : indeks esai yang dipakai melatih (subset idx_latih; hanya K5 yang membuang),
        P         : (len(idx), 5) sebaran target atas skor 1..5 (jumlah tiap baris 1),
        acuan_uji : (len(idx_uji),) skor acuan SENDIRI untuk esai uji (dipakai untuk QWK pada target sendiri),
        info      : dict ringkasan untuk log)
Fungsi `bangun` hanya boleh membaca skor penilai baris idx_latih untuk MEMBENTUK target latih; skor esai uji
hanya dipakai untuk acuan_uji (skor acuan esai uji itu sendiri). Ini diperiksa oleh uji_kontrol/uji_target.py.
"""
import importlib

_MODUL = {
    "K2": "k2_rerata",
    "K3": "k3_tertinggi",
    "K4": "k4_terendah",
    "K5": "k5_buang_selisih",
    "K6": "k6_rasch_keketatan",
}


def muat(kode):
    return importlib.import_module("target." + _MODUL[kode])
