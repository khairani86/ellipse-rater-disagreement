# Paket kode RQ2: cara membentuk skor acuan dari dua penilai dan kinerja AES

Menjawab RQ2 pada 6.478 esai resmi ELLIPSE (ber-`text_id_kaggle`, Overall > 0 pada kedua penilai): apakah cara membentuk skor acuan dari dua skor penilai (K2 sampai K6) mengubah QWK DeBERTa-v3-base, pada skor acuannya sendiri, pada rerata dua penilai (hasil utama), dan dengan PRMSE.

Rancangan dan rujukannya ada di Doc (bagian Rujukan RQ2). Paket ini **satu berkas per K dan satu berkas per pembanding**, tiap-tiapnya bisa dijalankan dan diuji sendiri.

## 0. Prinsip

- Setiap metode K dan pembanding mencatat di dalam berkasnya sendiri `RUJUKAN` (apa yang diambil dari makalah) dan `PILIHAN_KITA` (apa yang bukan dari rujukan).
- Setiap nilai di `config.json` punya cap di `sumber`: `rujukan`, `pilihan kita`, atau `BELUM ditetapkan`. `uji_kontrol/uji_konfigurasi.py` menolak kunci yang tidak bercap.
- Resep pelatihan sama untuk semua K dan tidak disetel per metode. Urutan esai dan inisialisasi kepala sama untuk semua metode pada (seed, lipatan) yang sama, jadi beda antar metode berasal dari targetnya, bukan dari kebisingan acak.
- Esai uji sama untuk semua metode (K5 hanya membuang esai dari data LATIH).

## 1. Isi paket

```
config.json           semua pengaturan + cap sumber tiap nilai
common.py             data resmi, lipatan per seed, kisi pembulatan, QWK, PRMSE, batas kappa, Nadeau-Bengio, Holm
mfrm_fast.py          model many-facet Rasch (MMLE, gradien analitik), dipakai K6
target/               SATU BERKAS PER K
  k2_rerata.py  k3_tertinggi.py  k4_terendah.py  k5_buang_selisih.py  k6_rasch_keketatan.py
baseline/             SATU BERKAS PER PEMBANDING (target K2)
  b0_konstanta.py  b1_ridge_tfidf.py  b2_bert_standar.py  b3_deberta_regresi.py
umum/                 token_cache.py  latih.py (pelatihan saraf)  evaluasi.py  uji_statistik.py  io_hasil.py
jalankan_k.py         menjalankan satu K (tahap target, tahap latih)
jalankan_baseline.py  menjalankan satu pembanding
ringkas.py            ukuran per lipatan, ringkasan, uji terhadap K2
check_env.py          memeriksa GPU dan paket
uji_kontrol/          uji kontrol (CPU) dan pemeriksa hasil uji coba GPU
```

## 2. Pemasangan (sekali)

Butuh Python 3.10 atau lebih baru. RTX 50xx (Blackwell) memerlukan PyTorch build CUDA 12.8 (`cu128`) atau lebih baru; GPU lain pakai perintah dari pytorch.org. Perlu internet untuk mengunduh `microsoft/deberta-v3-base` dan `bert-base-uncased`.

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Salin berkas ELLIPSE mentah ke `data/ellipse_raw.csv`. Kolom yang dipakai: `text_id_kaggle, Text, Rater_1, Rater_2, Overall_1, Overall_2`.

```
python check_env.py
```

## 3. Uji kontrol (CPU, tanpa GPU dan tanpa unduhan)

```
python uji_kontrol/jalankan_semua_uji.py
```

Isinya (tiap berkas bisa dijalankan sendiri):

| Berkas | Yang diperiksa |
|---|---|
| `uji_konfigurasi.py` | tiap kunci config bercap sumber; tiap K dan pembanding mencatat rujukan dan pilihan kita |
| `uji_data.py` | 6.478 esai, 4 dibuang karena skor 0, 18 esai selisih ≥ 2, lipatan seimbang dan berbeda tiap seed, faktor Nadeau-Bengio 2,18 |
| `uji_target.py [K]` | per K, pada data asli: bentuk target, jumlah baris 1, deterministik, **tidak bocor** (skor esai uji diacak, target latih harus sama), sifat khas tiap K (lihat bagian 5) |
| `uji_baseline.py` | B0 dan B1: nilai yang diharapkan, tidak bocor, alpha hanya dari data latih, IDF hanya dari data latih |
| `uji_evaluasi.py` | QWK dicocokkan dengan scikit-learn; PRMSE dan batas kappa dicocokkan pada simulasi dengan jawaban diketahui; Nadeau-Bengio dan Holm dicocokkan pada contoh hitung |
| `uji_ujung_ke_ujung.py` | seluruh jalur (token, target, pelatihan, prediksi, ringkasan, uji) dengan model DeBERTa dan BERT mini berbobot acak; menguji kode, bukan mutu model |

## 4. Uji coba per K dan per pembanding (GPU)

Tiap uji coba memakai satu (seed 0, lipatan 0), sekitar 10 menit per K. Urutannya sama untuk semua K; contoh K3:

```
python umum/token_cache.py deberta
python jalankan_k.py --k K3 --tahap target
python jalankan_k.py --k K3 --tahap latih --ukur
python jalankan_k.py --k K3 --tahap latih --seed 0 --fold 0
python uji_kontrol/periksa_hasil.py --k K3 --seed 0 --fold 0
```

`--ukur` mencetak kecepatan, memori puncak GPU, dan perkiraan waktu 1 pelatihan, lalu keluar. `periksa_hasil.py` memastikan model tidak runtuh, loss turun, loss tidak di bawah entropi target, keluaran berbentuk benar, dan target yang dipakai benar. Itu pemeriksaan teknis, bukan hasil penelitian.

Pembanding:

```
python jalankan_baseline.py --b B0
python jalankan_baseline.py --b B1 --seed 0 --fold 0
python umum/token_cache.py bert
python jalankan_baseline.py --b B2 --seed 0 --fold 0
python jalankan_baseline.py --b B3 --seed 0 --fold 0
python uji_kontrol/periksa_hasil.py --b B3 --seed 0 --fold 0
```

Kalau memori GPU kurang, tambahkan `--micro-batch 2` (batch efektif tetap 16 untuk DeBERTa dan 32 untuk BERT).

## 5. Menjalankan penuh

Tiap perintah melewati hasil yang sudah ada, jadi boleh dihentikan dan dilanjutkan.

```
python jalankan_k.py --k semua --tahap target
python jalankan_k.py --k semua --tahap latih
python jalankan_baseline.py --b semua
python ringkas.py
```

Perkiraan waktu: 75 pelatihan K (5 K × 15) + 15 pelatihan B3 = 90 pelatihan DeBERTa. Dari log lama (13,1 menit untuk 7.087 esai latih) diperkirakan sekitar 9,6 menit pada 5.182 esai latih, jadi sekitar 14,4 jam; B2 (BERT) 15 pelatihan belum terukur (paling lama +2,4 jam). B0 detik, B1 sekitar 15 detik per lipatan. Angka yang pasti diberikan `--ukur`.

Keluaran: `preds/` (prediksi esai uji), `hasil/metrik_per_lipatan.csv`, `hasil/ringkasan.md`, `hasil/uji_vs_K2.csv`.

## 6. Apa yang dari rujukan dan apa yang pilihan kita

| Bagian | Dari rujukan | Pilihan kita (bukan dari rujukan) |
|---|---|---|
| K2 rerata | Uto (2026); Shabaruddin & Sim (2026): target = rerata dua penilai | pemetaan ke sebaran dua kelas bersebelahan (kepala 5 logit) |
| K3, K4 tertinggi, terendah | Uto (2026), Lampiran B: aturan terkecil/terbesar pada skor gabungan ASAP asli | memakainya sebagai target latih pada ELLIPSE (Uto sendiri memakai rerata) |
| K5 buang selisih besar | Uma dkk. (2021), Bagian 3.2: melatih pada subset berkesepakatan tinggi | ambang selisih ≥ 2; esai uji tidak dibuang |
| K6 Rasch keketatan | Uto & Okano (2021), Persamaan 9-11: skor esai dari model Rasch, prediksi = rerata atas semua penilai; Wind dkk. (2018): alasan, bukan metode | hanya keketatan; MMLE + EAP (bukan Bayes penuh); target = sebaran lima skor (bukan regresi pada θ); rerata atas penilai yang punya data latih |
| Model K | DeBERTa-v3-base, lr 2e-5, batch 16, 512 token (Choi dkk. 2026; Zu & Badola 2026; Fan & Yun 2026) | 3 epoch, kepala 5 logit + cross-entropy lunak, pooling rata-rata, warmup, weight decay, dropout, gradient clip, bf16 |
| B0 konstanta | Shabaruddin & Sim (2026), B0 | target K2 |
| B1 Ridge TF-IDF | Shabaruddin & Sim (2026), T2a: TF-IDF kata 1-2 gram; alpha dipilih di 3 lipatan dalam hanya dari data latih; transformasi ditaksir hanya dari data latih | `min_df` = 2, kisi alpha, kriteria pemilihan alpha (makalahnya tidak menyebut) |
| B2 BERT standar | Uto (2026): "BERT dengan arsitektur standar"; Uto & Okano (2021): BERT base, [CLS] → linear + sigmoid, MSE pada skor [0,1], batch 32, epoch maks 3, Adam | bert-base-uncased; lr 2e-5; AdamW; 3 epoch tetap tanpa penghentian dini; anggapan bahwa "arsitektur standar" Uto (2026) sama dengan Uto & Okano |
| B3 DeBERTa regresi | Fan & Yun (2026): kepala regresi, MSE, skor ke [0,1]; Zu & Badola (2026): regresi MSE | resep sama dengan K; tanpa sigmoid; tanpa LoRA |
| Lipatan dan seed | Shabaruddin & Sim (2026): 5 lipatan × 3 ulangan, seed berbeda, 15 estimasi | pembagian lipatan acak ulang tiap seed tanpa lapisan (mereka melapisi menurut kuintil skor gabungan) |
| Evaluasi | Uto (2026): QWK dibulatkan dan dipotong, batas κ_max, κ_HL; Shabaruddin & Sim: QWK pada kisi setengah poin, metrik lain tanpa dibulatkan; Loukina dkk. (2020): PRMSE | **kisi pembulatan QWK belum ditetapkan**: bulat dan setengah poin dua-duanya dihitung |
| Inferensi | Shabaruddin & Sim (2026): Nadeau-Bengio dan Holm-Bonferroni | keluarga Holm = 4 pembanding (K3 sampai K6 terhadap K2); rumus dari penjelasan makalah itu, makalah asli Nadeau-Bengio dan Holm belum dibuka |

## 7. Hal terbuka yang perlu Anda ketahui

1. **K6 tidak memakai JMLE seperti RQ1c.** JMLE tidak memberi θ berhingga pada esai berskor 5-5 atau 1-1 (40 esai di lipatan latih pertama). K6 memakai MMLE dengan θ ~ N(μ, σ²) dan EAP; ini lebih dekat ke Uto & Okano (EAP) daripada JMLE. Kalimat K6 di Doc sudah disesuaikan.
2. **Skor acuan sendiri K5 = rerata dua penilai** (sama dengan K2). Itu akibat mekanis: K5 hanya mengubah data latih, bukan definisi skor. Maka QWK "pada skor acuan sendiri" K5 pada dasarnya sama dengan QWK pada rerata.
3. **K6 memampatkan skala**: simpangan baku nilai harapan target K6 sekitar 0,48 dibanding 0,64 pada K2 (penyusutan EAP dan rerata atas penilai).
4. Kisi pembulatan QWK belum ditetapkan (lihat tabel). `kisi_ringkasan` di config hanya mengatur kolom mana yang tampil lebih dulu.
5. Esai yang lebih panjang dari 512 token terpotong; persentasenya dicetak oleh `umum/token_cache.py`. Catat di keterbatasan.
6. Jumlah penilai dalam himpunan resmi 27, sedangkan makalah korpus menyebut 26; belum terselesaikan.
