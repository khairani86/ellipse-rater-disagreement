"""io_hasil.py - baca/tulis berkas target dan prediksi dengan nama yang sama untuk semua metode.

work/target/<K>_s<seed>_f<lipatan>.npz  : idx (esai latih yang dipakai), P (sebaran target), idx_uji, acuan_uji  (+ .json: ringkasan)
preds/<kode>/<kode>_s<seed>_f<lipatan>.npz : row_id (esai uji), pred (prediksi kontinu), probs (K2-K6; None untuk pembanding), acuan_uji (+ .json: log)
Mode --smoke menulis ke preds/smoke/<kode>/ dan work/target_smoke/ agar tidak bercampur dengan hasil sungguhan.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C  # noqa: E402


def _nama(kode, seed, fold):
    return f"{kode}_s{seed}_f{fold}"


def jalur_target(kode, seed, fold, smoke=False):
    return C.wpath("work", "target_smoke" if smoke else "target", _nama(kode, seed, fold) + ".npz")


def jalur_pred(kode, seed, fold, smoke=False):
    return C.wpath("preds", *(("smoke",) if smoke else ()), kode, _nama(kode, seed, fold) + ".npz")


def _json_aman(o):
    if isinstance(o, dict):
        return {k: _json_aman(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_json_aman(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def tulis(fn, arrays, info):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez(fn, **{k: v for k, v in arrays.items() if v is not None})
    with open(fn.replace(".npz", ".json"), "w", encoding="utf-8") as f:
        json.dump(_json_aman(info), f, indent=1, ensure_ascii=False)


def simpan_target(kode, seed, fold, hasil, idx_uji, smoke=False):
    tulis(jalur_target(kode, seed, fold, smoke),
          dict(idx=np.asarray(hasil["idx"]), P=np.asarray(hasil["P"]), idx_uji=np.asarray(idx_uji), acuan_uji=np.asarray(hasil["acuan_uji"])),
          hasil["info"])


def muat_target(kode, seed, fold, smoke=False):
    fn = jalur_target(kode, seed, fold, smoke)
    if not os.path.exists(fn):
        raise SystemExit(f"Target belum dibangun: {fn}\nJalankan dulu:  python jalankan_k.py --k {kode} --tahap target")
    z = np.load(fn)
    return {k: z[k] for k in z.files}


def simpan_pred(kode, seed, fold, out, acuan_uji, smoke=False):
    tulis(jalur_pred(kode, seed, fold, smoke),
          dict(row_id=np.asarray(out["idx_uji"]), pred=np.asarray(out["pred"]), probs=out.get("probs"), acuan_uji=np.asarray(acuan_uji)),
          out["log"])


def muat_pred(kode, seed, fold, smoke=False):
    fn = jalur_pred(kode, seed, fold, smoke)
    if not os.path.exists(fn):
        return None
    z = np.load(fn)
    return {k: z[k] for k in z.files}
