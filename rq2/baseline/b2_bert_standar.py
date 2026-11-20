"""B2 - BERT standar: [CLS] -> linear -> sigmoid, MSE pada skor yang dinormalkan ke [0,1], target K2.

Rujukan : Uto (2026) memakai "BERT-based AES model with a standard architecture" (QWK dengan prediksi dan skor sasaran dibulatkan dan dipotong);
          rinciannya (Uto & Okano 2021, Bagian III-D dan eksperimen): BERT berukuran base, keluaran [CLS] ke lapisan linear dengan aktivasi sigmoid,
          skor dinormalkan ke [0,1], galat kuadrat rerata (MSE), mini-batch 32, epoch maksimum 3, optimizer Adam.
Catatan  : Uto (2026) hanya menyebut "arsitektur standar"; kita menganggapnya sama dengan Uto & Okano (2021), penulis yang sama (belum kita periksa rujukan [22] Uto 2026).
Pilihan kita: bert-base-uncased, lr 2e-5 (sama dengan K), AdamW dengan weight decay 0,01 (Uto & Okano: Adam), pemanasan 10% lalu peluruhan linear,
          dropout 0,1, pemotongan gradien 1,0, bf16, epoch tetap 3 tanpa penghentian dini (Uto & Okano: maksimum 3), 512 token.
"""


import common as C
from umum import latih, token_cache

KODE = "B2"
NAMA = C.NAMA["B2"]
RUJUKAN = ["Uto (2026), arXiv 2604.19131 (BERT dengan arsitektur standar)",
           "Uto & Okano (2021), IEEE TLT 14(6), 763-776 (BERT base, [CLS], linear + sigmoid, MSE, batch 32, epoch maks 3, Adam)"]
PILIHAN_KITA = ["bert-base-uncased", "lr 2e-5, AdamW wd 0,01, pemanasan 10%", "dropout 0,1, gradient clip 1,0, bf16", "epoch tetap 3, tanpa penghentian dini"]


def resep(cfg):
    r = {k: cfg[k] for k in ("lr_backbone", "lr_head", "weight_decay", "warmup_ratio", "dropout", "grad_clip", "bf16", "grad_ckpt", "eval_batch")}
    r.update(epochs=cfg["bert_epochs"], micro_batch=cfg["bert_micro_batch"], accum=cfg["bert_accum"])
    return r


def jalankan(df, idx_latih, idx_uji, seed, lipatan, cfg, opsi):
    tok = token_cache.muat("bert")
    out = latih.latih_prediksi(tok, cfg["model_bert"], "regresi", "cls", True, idx_latih, df.rerata.values[idx_latih], idx_uji,
                               seed, lipatan, resep(cfg), opsi, "B2")
    return out
