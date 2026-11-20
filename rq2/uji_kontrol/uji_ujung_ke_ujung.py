"""uji_ujung_ke_ujung.py - seluruh jalur kode (token, target, pelatihan, prediksi, evaluasi, uji) dengan MODEL MINI lokal (bobot acak,
kosakata kecil): tanpa unduhan dan tanpa GPU. Yang diuji adalah JALUR KODE, bukan mutu model. Memakai folder terpisah (work_uji_ujung dst.).

Pemakaian:  python uji_kontrol/uji_ujung_ke_ujung.py        (sekitar 3-6 menit di CPU)
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import _bantu as B

TAG = "_uji_ujung"
os.environ["RQ2_TAG"] = TAG
import common as C  # noqa: E402

env = dict(os.environ, RQ2_TAG=TAG)
tmp = tempfile.mkdtemp(prefix="rq2_mini_")
for top in ("work", "preds", "hasil"):
    shutil.rmtree(C.wpath(top), ignore_errors=True)


def sh(*args, cek=True):
    r = subprocess.run([sys.executable, *args], cwd=B.ROOT, env=env, capture_output=True, text=True, errors="replace")
    if cek and r.returncode != 0:
        print(r.stdout[-2500:], r.stderr[-2500:])
        raise SystemExit(f"gagal: {' '.join(args)}")
    return r


df = C.muat_baris()
cfg0 = C.config()

# ---------------------------------------------------------------- model mini: DeBERTa (SentencePiece kecil) dan BERT (WordPiece karakter)
import sentencepiece as spm                                                       # noqa: E402
from transformers import (BertConfig, BertModel, BertTokenizer, DebertaV2Config, DebertaV2Model, DebertaV2Tokenizer)  # noqa: E402

txt = os.path.join(tmp, "korpus.txt")
with open(txt, "w", encoding="utf-8") as f:
    for t in df.Text.astype(str).head(2000):
        f.write(" ".join(t.split()) + "\n")
spm.SentencePieceTrainer.train(input=txt, model_prefix=os.path.join(tmp, "spm"), vocab_size=400, model_type="unigram", pad_id=0, unk_id=1, bos_id=2,
                               eos_id=3, pad_piece="[PAD]", unk_piece="[UNK]", bos_piece="[CLS]", eos_piece="[SEP]", user_defined_symbols=["[MASK]"],
                               character_coverage=1.0)
tok_d = DebertaV2Tokenizer(vocab_file=os.path.join(tmp, "spm.model"), do_lower_case=False)
cfg_d = DebertaV2Config(vocab_size=400, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, max_position_embeddings=512,
                        relative_attention=True, pos_att_type=["p2c", "c2p"], position_buckets=64, norm_rel_ebd="layer_norm", share_att_key=True,
                        position_biased_input=False, max_relative_positions=-1)
dir_d = os.path.join(tmp, "mini_deberta")
DebertaV2Model(cfg_d).save_pretrained(dir_d)
tok_d.save_pretrained(dir_d)

huruf = list("abcdefghijklmnopqrstuvwxyz0123456789.,!?'\"-():;/")
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + huruf + ["##" + h for h in huruf]
with open(os.path.join(tmp, "vocab.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(vocab) + "\n")
tok_b = BertTokenizer(os.path.join(tmp, "vocab.txt"), do_lower_case=True)
cfg_b = BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, max_position_embeddings=512)
dir_b = os.path.join(tmp, "mini_bert")
BertModel(cfg_b).save_pretrained(dir_b)
tok_b.save_pretrained(dir_b)
env["RQ2_DEBERTA"], env["RQ2_BERT"] = dir_d, dir_b
os.environ["RQ2_DEBERTA"], os.environ["RQ2_BERT"] = dir_d, dir_b
print("model mini:", dir_d, dir_b)

# ---------------------------------------------------------------- 1. token
for kunci in ("deberta", "bert"):
    r = sh("umum/token_cache.py", kunci)
    z = np.load(C.wpath("work", f"tokens_{kunci}.npz"))
    B.periksa(f"token {kunci}: satu urutan per esai, id di bawah ukuran kosakata, panjang <= 512",
              len(z["offs"]) == len(df) + 1 and z["flat"].max() < (400 if kunci == "deberta" else len(vocab)) and np.diff(z["offs"]).max() <= 512)

# ---------------------------------------------------------------- 2. target semua K (smoke) pada dua lipatan
for fold in (0, 1):
    sh("jalankan_k.py", "--k", "semua", "--tahap", "target", "--smoke", "--seed", "0", "--fold", str(fold))
from umum import io_hasil  # noqa: E402
B.periksa("target: 5 K x 2 lipatan tersimpan", all(os.path.exists(io_hasil.jalur_target(k, 0, f, True)) for k in C.METODE for f in (0, 1)))

# ---------------------------------------------------------------- 3. latih semua K (CPU, kecil) pada dua lipatan
for fold in (0, 1):
    r = sh("jalankan_k.py", "--k", "semua", "--tahap", "latih", "--smoke", "--cpu", "--no-grad-ckpt", "--micro-batch", "4", "--seed", "0", "--fold", str(fold))
for k in C.METODE:
    z = io_hasil.muat_pred(k, 0, 0, True)
    ok = z is not None and z["probs"].shape == (32, 5) and np.allclose(z["probs"].sum(1), 1, atol=1e-4) and np.isfinite(z["probs"]).all()
    ok = ok and np.allclose(z["pred"], z["probs"] @ C.SKALA, atol=1e-6) and len(z["acuan_uji"]) == 32 and (z["pred"] > 1 - 1e-6).all() and (z["pred"] < 5 + 1e-6).all()
    B.periksa(f"{k}: prediksi smoke sah (32 esai, probabilitas berjumlah 1, nilai harapan = prediksi, dalam [1, 5])", ok)
z2, z3 = io_hasil.muat_pred("K2", 0, 0, True), io_hasil.muat_pred("K3", 0, 0, True)
B.periksa("K2 dan K3 pada (seed, lipatan) sama memakai esai uji yang sama tetapi target berbeda -> prediksi berbeda",
          np.array_equal(z2["row_id"], z3["row_id"]) and not np.allclose(z2["pred"], z3["pred"]))
B.periksa("acuan_uji K3 pada berkas prediksi = skor tertinggi esai uji", np.array_equal(z3["acuan_uji"], np.maximum(df.x_a.values[z3["row_id"]], df.x_b.values[z3["row_id"]])))
sh("jalankan_k.py", "--k", "K2", "--tahap", "latih", "--smoke", "--cpu", "--no-grad-ckpt", "--micro-batch", "4", "--seed", "0", "--fold", "0", "--paksa")
B.periksa("K2: ulang dengan seed sama -> prediksi identik (deterministik)", np.allclose(io_hasil.muat_pred("K2", 0, 0, True)["pred"], z2["pred"], atol=1e-6))
r = sh("jalankan_k.py", "--k", "K6", "--tahap", "latih", "--smoke", "--cpu", "--seed", "1", "--fold", "0", cek=False)
B.periksa("tanpa target dibangun, tahap latih berhenti dengan petunjuk (bukan jatuh diam-diam)", r.returncode != 0 and "tahap target" in (r.stdout + r.stderr))
sh("jalankan_k.py", "--k", "K2", "--tahap", "target", "--seed", "0", "--fold", "0")           # target NYATA (bukan smoke) untuk uji --ukur
r = sh("jalankan_k.py", "--k", "K2", "--tahap", "latih", "--cpu", "--ukur", "--ukur-langkah", "3", "--no-grad-ckpt", "--micro-batch", "4", "--seed", "0", "--fold", "0")
B.periksa("mode --ukur K2 (target nyata, model mini, CPU) menampilkan kecepatan dan perkiraan waktu", "[ukur]" in r.stdout and "perkiraan 1 pelatihan" in r.stdout, r.stdout[-300:])
B.periksa("mode --ukur tidak menulis prediksi", io_hasil.muat_pred("K2", 0, 0, False) is None)

# ---------------------------------------------------------------- 4. pembanding
for fold in (0, 1):
    sh("jalankan_baseline.py", "--b", "B0", "--smoke", "--seed", "0", "--fold", str(fold))
    sh("jalankan_baseline.py", "--b", "B1", "--smoke", "--seed", "0", "--fold", str(fold))
    for b in ("B2", "B3"):
        sh("jalankan_baseline.py", "--b", b, "--smoke", "--cpu", "--no-grad-ckpt", "--micro-batch", "4", "--seed", "0", "--fold", str(fold))
for b in C.PEMBANDING:
    z = io_hasil.muat_pred(b, 0, 0, True)
    B.periksa(f"{b}: prediksi smoke berhingga, acuan_uji = rerata dua penilai (target K2)", z is not None and np.isfinite(z["pred"]).all()
              and np.allclose(z["acuan_uji"], df.rerata.values[z["row_id"]]))
z2s, z3s = io_hasil.muat_pred("B2", 0, 0, True), io_hasil.muat_pred("B3", 0, 0, True)
B.periksa("B2: keluaran lewat sigmoid -> prediksi dalam [1, 5]; B3: tanpa sigmoid -> prediksi berhingga",
          (z2s["pred"] >= 1 - 1e-9).all() and (z2s["pred"] <= 5 + 1e-9).all() and np.isfinite(z3s["pred"]).all())
B.periksa("B2 dan B3: tidak ada probabilitas (kepala regresi)", "probs" not in z2s.files if hasattr(z2s, "files") else "probs" not in z2s)
r = sh("jalankan_baseline.py", "--b", "B3", "--cpu", "--ukur", "--ukur-langkah", "3", "--no-grad-ckpt", "--micro-batch", "4", "--seed", "0", "--fold", "0")
B.periksa("pembanding B3: mode --ukur menampilkan kecepatan", "[ukur]" in r.stdout, r.stdout[-300:])

# ---------------------------------------------------------------- 5. ringkasan dan uji
r = sh("ringkas.py", "--smoke")
import pandas as pd  # noqa: E402
tab = pd.read_csv(C.wpath("hasil", "metrik_per_lipatan.csv"))
B.periksa("ringkas: 9 metode x 2 lipatan = 18 baris, tanpa NaN pada QWK dan PRMSE", len(tab) == 18 and tab[["qwk_rerata_bulat", "qwk_rerata_setengah", "prmse"]].notna().all().all(), str(len(tab)))
uji = pd.read_csv(C.wpath("hasil", "uji_vs_K2.csv"))
B.periksa("ringkas: uji terhadap K2 untuk K3-K6 pada tiap ukuran (4 baris per ukuran, p Holm di [0, 1])",
          len(uji) == 4 * uji.ukuran.nunique() and uji.p_holm.between(0, 1).all(), str(len(uji)))
B.periksa("ringkas.md dan ringkasan.csv ditulis", os.path.exists(C.wpath("hasil", "ringkasan.md")) and os.path.exists(C.wpath("hasil", "ringkasan.csv")))

# ---------------------------------------------------------------- 6. fungsi kecil
import torch  # noqa: E402
from umum import latih  # noqa: E402
lg = torch.randn(6, 5)
one = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3, 4, 2]), 5).float()
B.periksa("soft_ce dengan target satu kelas = cross-entropy biasa", abs(float(latih.soft_ce(lg, one)) - float(torch.nn.functional.cross_entropy(lg, torch.tensor([0, 1, 2, 3, 4, 2])))) < 1e-6)
sof = torch.tensor([[0.5, 0.5, 0, 0, 0]] * 6)
B.periksa("soft_ce dengan target 0,5/0,5 = rerata dua cross-entropy", abs(float(latih.soft_ce(lg, sof)) - float(0.5 * torch.nn.functional.cross_entropy(lg, torch.zeros(6, dtype=torch.long)) + 0.5 * torch.nn.functional.cross_entropy(lg, torch.ones(6, dtype=torch.long)))) < 1e-6)
B.periksa("skor <-> [0,1]: 1 -> 0, 5 -> 1, dan balik", np.allclose(latih.skor_ke_norm([1, 3, 5]), [0, 0.5, 1]) and np.allclose(latih.norm_ke_skor([0, 0.5, 1]), [1, 3, 5]))
d16 = os.path.join(tmp, "mini16")
DebertaV2Model(cfg_d).half().save_pretrained(d16)
enc = C.muat_encoder(d16)
B.periksa("muat_encoder: checkpoint fp16 dimuat sebagai fp32", {p.dtype for p in enc.parameters()} == {torch.float32})

shutil.rmtree(tmp, ignore_errors=True)
B.akhiri("uji_ujung_ke_ujung")
