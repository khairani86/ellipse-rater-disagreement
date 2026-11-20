"""_bantu.py - pencatat hasil pemeriksaan yang dipakai semua berkas uji_kontrol."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_gagal = []
_jumlah = [0]


def periksa(nama, kondisi, detail=""):
    _jumlah[0] += 1
    ok = bool(kondisi)
    if ok:
        ket = f"  [{detail}]" if detail and detail != "[]" and len(detail) <= 70 else ""
    else:
        ket = f"  ({detail})" if detail else ""
    print(("[OK]    " if ok else "[GAGAL] ") + nama + ket, flush=True)
    if not ok:
        _gagal.append(nama)
    return ok


def akhiri(judul):
    print(f"\n{judul}: {_jumlah[0] - len(_gagal)} dari {_jumlah[0]} pemeriksaan lulus.")
    if _gagal:
        print("GAGAL:", *_gagal, sep="\n  - ")
        sys.exit(1)
    print("SEMUA LULUS")
