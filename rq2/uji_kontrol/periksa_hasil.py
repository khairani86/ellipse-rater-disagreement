"""periksa_hasil.py - pemeriksaan teknis SATU hasil uji coba GPU (satu K atau satu pembanding, satu seed dan lipatan).
Ini BUKAN uji hipotesis: hanya memastikan pelatihan tidak rusak (model tidak runtuh, loss turun, keluaran berbentuk benar, target yang dipakai benar).

Pemakaian:  python uji_kontrol/periksa_hasil.py --k K3 --seed 0 --fold 0
            python uji_kontrol/periksa_hasil.py --b B3 --seed 0 --fold 0
"""
import argparse
import json

import numpy as np

import _bantu as B
import common as C
from umum import evaluasi, io_hasil

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--k", choices=C.METODE)
g.add_argument("--b", choices=C.PEMBANDING)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
kode = a.k or a.b
cfg = C.config()
df = C.muat_baris()
z = io_hasil.muat_pred(kode, a.seed, a.fold, a.smoke)
if z is None:
    raise SystemExit(f"Belum ada prediksi {kode} s{a.seed} f{a.fold}: {io_hasil.jalur_pred(kode, a.seed, a.fold, a.smoke)}")
with open(io_hasil.jalur_pred(kode, a.seed, a.fold, a.smoke).replace(".npz", ".json"), encoding="utf-8") as f:
    log = json.load(f)
tr, te = C.bagi(len(df), a.seed, a.fold, cfg)
pred = z["pred"]

B.periksa("esai uji = esai uji lipatan itu (urutan dan jumlah)", set(z["row_id"]) == set(te) if not a.smoke else set(z["row_id"]) <= set(te), f"{len(z['row_id'])} esai")
B.periksa("prediksi berhingga", np.isfinite(pred).all())
B.periksa("model tidak runtuh: SD prediksi antar esai > 0,05" + (" [dilewati pada smoke]" if a.smoke else ""), a.smoke or pred.std() > 0.05, f"SD {pred.std():.3f}")
B.periksa("prediksi sebagian besar masuk akal: 98% di dalam [0,5; 5,5]", np.mean((pred > 0.5) & (pred < 5.5)) >= 0.98, f"rentang {pred.min():.2f} sampai {pred.max():.2f}")
if kode in C.METODE:
    t = io_hasil.muat_target(kode, a.seed, a.fold, a.smoke)
    B.periksa("jumlah esai latih pada log = jumlah baris target", log["n_latih"] == len(t["idx"]) or a.smoke, f"{log['n_latih']}")
    B.periksa("acuan_uji pada prediksi = acuan_uji pada target", np.allclose(z["acuan_uji"], t["acuan_uji"][:len(z["acuan_uji"])]))
    B.periksa("probabilitas berjumlah 1 dan nilai harapan = prediksi", np.allclose(z["probs"].sum(1), 1, atol=1e-4) and np.allclose(z["probs"] @ C.SKALA, pred, atol=1e-5))
    # loss minimum teoretis = entropi target; loss tidak boleh di bawahnya
    P = t["P"][:log["n_latih"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropi = float(-np.where(P > 0, P * np.log(P), 0).sum(1).mean())
    el = log["epoch_loss"]
    B.periksa("loss tiap epoch turun (epoch terakhir < epoch pertama)", len(el) < 2 or el[-1] < el[0], str([round(x, 3) for x in el]) + (" (1 epoch: dilewati)" if len(el) < 2 else ""))
    B.periksa("loss terakhir tidak di bawah entropi target (batas bawah teoretis) dikurangi galat kecil", el[-1] > entropi - 0.02, f"loss {el[-1]:.3f}, entropi target {entropi:.3f}")
else:
    el = log.get("epoch_loss", [])
    if el:
        B.periksa("loss tiap epoch turun", len(el) < 2 or el[-1] < el[0], str([round(x, 4) for x in el]) + (" (1 epoch: dilewati)" if len(el) < 2 else ""))
m = evaluasi.metrik(df, z["row_id"], pred, z["acuan_uji"])
print(f"\nUkuran pada lipatan ini ({kode}, seed {a.seed}, lipatan {a.fold}) - hanya untuk dilihat, bukan hasil akhir:")
for k_, v in m.items():
    print(f"  {k_:22s} {C.fmt(v)}")
B.periksa("QWK pada rerata dua penilai di atas konstanta (B0 = 0), kisi setengah" + (" [dilewati pada smoke]" if a.smoke else ""), a.smoke or m["qwk_rerata_setengah"] > 0.1, C.fmt(m["qwk_rerata_setengah"]))
B.periksa("korelasi dengan rerata dua penilai > 0,3" + (" [dilewati pada smoke]" if a.smoke else ""), a.smoke or m["korelasi_rerata"] > 0.3, C.fmt(m["korelasi_rerata"]))
print("\nLog pelatihan:", {k_: v for k_, v in log.items() if k_ not in ("epoch_loss", "mse_dalam")})
B.akhiri("periksa_hasil")
