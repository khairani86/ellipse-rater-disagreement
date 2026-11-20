"""B3 - DeBERTa-v3-base dengan kepala regresi (satu keluaran), MSE pada skor yang dinormalkan ke [0,1], target K2.

Rujukan : Fan & Yun (2026), arXiv 2601.16724: DeBERTa-v3-base, kepala regresi, MSE, skor dinormalkan ke [0,1] (ELLIPSE gabungan);
          Zu & Badola (2026), AIME-Con 490-496: DeBERTa-v3-base, regresi dengan MSE, dibulatkan untuk QWK (data bicara ETS, bukan ELLIPSE).
Pilihan kita: resep pelatihan SAMA dengan K2-K6 (lr 2e-5, batch 16, 3 epoch, pooling rata-rata token, dropout 0,1, ...), agar yang berbeda
          dari K2 hanya kepala dan fungsi galatnya. Tanpa sigmoid (kedua rujukan tidak menyebutnya). Fan & Yun memakai LoRA, lr 5e-5, 5 epoch;
          Zu & Badola lr 1e-5, sampai 4 epoch; kita tidak menirunya.
"""
import common as C
from umum import latih, token_cache

KODE = "B3"
NAMA = C.NAMA["B3"]
RUJUKAN = ["Fan & Yun (2026), arXiv 2601.16724", "Zu & Badola (2026), AIME-Con, 490-496"]
PILIHAN_KITA = ["resep sama dengan K2-K6 (hanya kepala dan galat yang berbeda)", "tanpa sigmoid", "tanpa LoRA"]


def resep(cfg):
    return {k: cfg[k] for k in ("epochs", "micro_batch", "accum", "lr_backbone", "lr_head", "weight_decay", "warmup_ratio",
                                "dropout", "grad_clip", "bf16", "grad_ckpt", "eval_batch")}


def jalankan(df, idx_latih, idx_uji, seed, lipatan, cfg, opsi):
    tok = token_cache.muat("deberta")
    return latih.latih_prediksi(tok, cfg["model_deberta"], "regresi", "rata", False, idx_latih, df.rerata.values[idx_latih], idx_uji,
                                seed, lipatan, resep(cfg), opsi, "B3")
