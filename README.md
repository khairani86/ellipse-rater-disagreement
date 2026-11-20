# ellipse-rater-disagreement

Code for a study of two-rater disagreement and reference-score construction for automated essay scoring (AES) on the ELLIPSE corpus: rater agreement, many-facet Rasch rater severity, and five ways to turn two rater scores into one reference score for DeBERTa-v3-base.

Kode untuk makalah "Ketidaksepakatan Dua Penilai dan Pembentukan Skor Acuan pada Penilaian Esai Otomatis pada Korpus ELLIPSE" (Khairani Djahara, Umi Laili Yuhana, Berlian Al Kindhi, Morteza Saberi). Makalah: [tautan akan ditambahkan].

## Isi

| Folder | Isi | Perangkat keras |
|---|---|---|
| `rq1/` | RQ1: kesepakatan dua penilai (RQ1a), pada skor berapa sulit sepakat (RQ1b), keketatan penilai dengan many-facet Rasch (RQ1c) | CPU, beberapa menit per skrip |
| `rq2/` | RQ2: lima skor acuan (K2 sampai K6) dan empat pembanding (B0 sampai B3), 5 lipatan × 3 seed, uji Nadeau–Bengio dengan koreksi Holm | GPU 8 GB (RTX 5060 pada kami), sekitar 21,5 jam |

Tiap folder punya `README.md` sendiri dengan pemasangan dan perintah lengkap. Jalankan perintah di dalam folder itu.

## Data

Kode ini **tidak menyertakan data**. Unduh korpus ELLIPSE (skor mentah per penilai, 8.890 baris) dari https://github.com/scrosseye/ELLIPSE-Corpus, lalu letakkan sebagai `rq1/data/ellipse_raw.csv` dan `rq2/data/ellipse_raw.csv`. Kode memakai 6.478 esai: baris ber-`text_id_kaggle` (himpunan resmi korpus) dengan skor holistik (Overall) lebih dari 0 pada kedua penilai.

## Urutan menjalankan

```
cd rq1
python rq1a_kesepakatan.py
python rq1b_tingkat_skor.py
python rq1c_keketatan.py
cd ..\rq2
python check_env.py
```

Perintah RQ2 selanjutnya (uji kontrol, K2 sampai K6, pembanding, ringkasan) ada di `rq2/README.md`.

## Angka yang harus muncul

Bila kode berjalan benar pada data yang sama, angka berikut harus terlihat (QWK terhadap rerata dua skor holistik, rerata 15 estimasi; kisi bulat / kisi setengah poin):

| Skor acuan | Kisi bulat | Kisi setengah poin |
|---|---|---|
| K2 rerata | 0,640 | 0,740 |
| K3 tertinggi | 0,647 | 0,668 |
| K4 terendah | 0,558 | 0,717 |
| K5 rerata tanpa esai berselisih ≥ 2 | 0,649 | 0,735 |
| K6 disesuaikan keketatan penilai | 0,571 | 0,689 |

RQ1a pada 6.478 esai: dua penilai memberi skor holistik sama persis pada 3.833 esai (59,2%), QWK 0,591.

RQ1 memakai seed tetap (20261115), jadi angkanya berulang persis. Pelatihan saraf RQ2 dapat sedikit berbeda antar GPU dan versi PyTorch, jadi selisih kecil pada angka RQ2 wajar.

## Lisensi

Kode: MIT (lihat `LICENSE`). Data ELLIPSE tunduk pada ketentuan pemiliknya.
