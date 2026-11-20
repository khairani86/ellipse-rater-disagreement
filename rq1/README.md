# Paket kode RQ1: ketidaksepakatan dua penilai pada ELLIPSE

Menjawab RQ1 pada 6.478 esai resmi ELLIPSE (baris ber-`text_id_kaggle`, Overall > 0 pada kedua penilai). Tiga skrip, satu per pertanyaan. Waktu di bawah diukur pada CPU biasa tanpa GPU. Tiap skrip berdiri sendiri: hanya butuh numpy dan pandas, tanpa GPU, dan tidak memakai berkas lain di paket.

| Skrip | Pertanyaan | Keluaran (folder) | Perkiraan waktu |
|---|---|---|---|
| `rq1a_kesepakatan.py` | RQ1a: seberapa setuju dua penilai (QWK, ICC(1,1), persis sama, selisih ≥ 2, batas κ_HL dan κ_max) | `hasil_rq1a/` | sekitar 15 detik |
| `rq1b_tingkat_skor.py` | RQ1b: pada skor berapa mereka berbeda | `hasil_rq1b/` | beberapa detik |
| `rq1c_keketatan.py` | RQ1c: keketatan penilai atau kesulitan aspek (many-facet Rasch kredit parsial, JMLE) | `hasil_rq1c/` | sekitar 5 menit |

Setiap skrip mencetak ringkasan di layar dan menulis `ringkasan_*.md` serta berkas `.csv` di folder keluarannya. Selang kepercayaan RQ1a dan RQ1b: bootstrap 2.000 ulangan per pasangan penilai, seed 20261115. RQ1c mencetak pula pemeriksaan hitungan (semuanya harus `LOLOS`).

## 1. Pemasangan (sekali)

Butuh Python 3.10 atau lebih baru.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data

Salin berkas ELLIPSE mentah (skor per penilai, 8.890 baris) ke `data/ellipse_raw.csv`. Kolom yang dipakai: `Filename, text_id_kaggle, Text, Rater_1, Rater_2` dan skor `Overall`, `Cohesion`, `Syntax`, `Vocabulary`, `Phraseology`, `Grammar`, `Conventions` untuk tiap penilai (akhiran `_1` dan `_2`). Korpus ELLIPSE: https://github.com/scrosseye/ELLIPSE-Corpus

Himpunan utama = baris ber-`text_id_kaggle` (6.478 esai). Pemeriksaan kepekaan pada semua baris dan pada baris tanpa id itu ikut dijalankan otomatis.

## 3. Menjalankan

```
python rq1a_kesepakatan.py
python rq1b_tingkat_skor.py
python rq1c_keketatan.py
```

Folder keluaran lain atau berkas data di tempat lain:

```
python rq1a_kesepakatan.py D:\data\ellipse_raw.csv D:\hasil_rq1a
```

Mengganti himpunan utama ke semua baris (hanya untuk pemeriksaan): tambahkan `--utama semua`. RQ1c: `--rep N` mengatur jumlah ulangan uji pemulihan parameter (bawaan 3).

## 4. Apa yang dari rujukan dan apa yang pilihan kita

| Bagian | Dari rujukan | Pilihan kita |
|---|---|---|
| RQ1a | QWK, ICC, persis sama, bootstrap: Kim (2026), James (2026); batas κ_HL dan κ_max dari ICC: Uto (2026) | bootstrap 2.000 ulangan dan seed |
| RQ1b | persis sama dan selisih ≥ 2 menurut skor satu penilai dan menurut rerata: Kim (2026), Tabel 6 dan 7 | (tidak ada) |
| RQ1c | many-facet Rasch dengan JMLE dan perbandingan rentang keketatan penilai dengan rentang kesulitan kriteria: Kim (2026); penandaan sel penilai × aspek: Wu dkk. (2026); koreksi q: Benjamini & Hochberg (1995); kredit parsial: Masters (1982) | memakai kredit parsial; enam aspek tanpa Overall (Overall sebagai kriteria ke-7 hanya pemeriksaan kepekaan); rasio simpangan baku dan selang 95% dari 2.000 undian atas galat baku; sel dianalisis bila ≥ 30 pengamatan |

Model RQ1c, ditulis sama dengan Rumus (1) di laporan:

```
ln [ P(skor = k) / P(skor = k-1) ] = theta_i - s_r - d_j - tau_jk ,   k = 2, 3, 4, 5
```

`theta_i` kemampuan esai, `s_r` keketatan penilai (positif = lebih ketat), `d_j` kesulitan aspek (positif = lebih sulit), `tau_jk` ambang aspek. Batasan: rerata `s_r` = 0, rerata `d_j` = 0, jumlah `tau_jk` per aspek = 0. Esai yang seluruh skornya 1 atau seluruh skornya 5 dikeluarkan.
