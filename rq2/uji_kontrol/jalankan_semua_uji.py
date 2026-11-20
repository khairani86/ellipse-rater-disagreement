"""jalankan_semua_uji.py - menjalankan semua uji kontrol CPU berurutan (tanpa GPU, tanpa unduhan).
Pemakaian:  python uji_kontrol/jalankan_semua_uji.py          (sekitar 4-8 menit)
            python uji_kontrol/jalankan_semua_uji.py --cepat   (tanpa uji ujung-ke-ujung dengan model mini)
"""
import subprocess
import sys

import _bantu as B

cepat = "--cepat" in sys.argv
daftar = ["uji_konfigurasi.py", "uji_data.py", "uji_evaluasi.py", "uji_target.py", "uji_baseline.py"] + ([] if cepat else ["uji_ujung_ke_ujung.py"])
hasil = {}
for nama in daftar:
    print(f"\n===== {nama} =====", flush=True)
    r = subprocess.run([sys.executable, f"{B.ROOT}/uji_kontrol/{nama}"], cwd=B.ROOT)
    hasil[nama] = r.returncode == 0
print("\n===== RINGKASAN =====")
for k, v in hasil.items():
    print(("[LULUS] " if v else "[GAGAL] ") + k)
sys.exit(0 if all(hasil.values()) else 1)
