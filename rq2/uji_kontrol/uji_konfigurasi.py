"""uji_konfigurasi.py - (1) tiap kunci config.json punya cap asal (rujukan / pilihan kita / belum ditetapkan);
(2) tiap berkas K dan pembanding mencatat RUJUKAN dan PILIHAN_KITA; (3) kata yang dilarang tidak muncul di berkas paket."""
import glob
import os

import _bantu as B
import common as C
import baseline
import target

cfg = C.config()
sumber = cfg["sumber"]
kunci = [k for k in cfg if not k.startswith("_") and k != "sumber"]
B.periksa("tiap kunci config punya cap sumber", all(k in sumber for k in kunci), str([k for k in kunci if k not in sumber]))
B.periksa("tiap cap sumber merujuk kunci yang ada", all(k in cfg for k in sumber), str([k for k in sumber if k not in cfg]))
awal = {k: sumber[k].split(":")[0].split(" ")[0].lower() for k in sumber}
B.periksa("cap diawali 'rujukan', 'pilihan' atau 'belum'", all(a in ("rujukan", "pilihan", "belum") for a in awal.values()), str(awal))
B.periksa("kisi pembulatan ditandai belum ditetapkan", sumber["kisi_ringkasan"].upper().startswith("BELUM"))
B.periksa("batch efektif DeBERTa 16 dan BERT 32 (rujukan)", cfg["micro_batch"] * cfg["accum"] == 16 and cfg["bert_micro_batch"] * cfg["bert_accum"] == 32)

for k in C.METODE:
    m = target.muat(k)
    B.periksa(f"{k}: modul memuat KODE, RUJUKAN, PILIHAN_KITA, bangun", m.KODE == k and len(m.RUJUKAN) > 0 and isinstance(m.PILIHAN_KITA, list) and callable(m.bangun))
for b in C.PEMBANDING:
    m = baseline.muat(b)
    B.periksa(f"{b}: modul memuat KODE, RUJUKAN, PILIHAN_KITA, jalankan", m.KODE == b and len(m.RUJUKAN) > 0 and isinstance(m.PILIHAN_KITA, list) and callable(m.jalankan))

kata = "bench" + "mark"                                           # ditulis terpisah agar berkas ini tidak menandai dirinya sendiri
ada = []
for fn in glob.glob(os.path.join(B.ROOT, "**", "*"), recursive=True):
    if os.path.isfile(fn) and fn.endswith((".py", ".md", ".json", ".txt")) and os.sep + "data" + os.sep not in fn and os.sep + "work" not in fn:
        with open(fn, encoding="utf-8", errors="replace") as f:
            if kata in f.read().lower():
                ada.append(os.path.relpath(fn, B.ROOT))
B.periksa("kata terlarang tidak ada di berkas paket", not ada, str(ada))
B.akhiri("uji_konfigurasi")
