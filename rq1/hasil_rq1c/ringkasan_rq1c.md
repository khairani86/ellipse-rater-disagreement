# RQ1c: keketatan penilai dibanding kesulitan aspek

Model many-facet Rasch kredit parsial, JMLE. Keketatan positif = lebih ketat; kesulitan positif = lebih sulit. Rasio dihitung dari rentang (seperti Kim) dan dari simpangan baku.

Himpunan utama: Baris ber-text_id_kaggle (himpunan resmi).

## Konteks: Kim (2026), bukan hasil kita

Rentang keketatan penilai 1,003 logit (42 penilai), pemisahan 7,33, keandalan 0,982; rentang kesulitan 9 kriteria sekitar 3,09 logit, dengan kriteria holistik (+2,108) sebagai yang tersulit. Tanpa holistik, rentang 8 kriteria analitik = 0,569 - (-0,981) = 1,550; rentang penilai : rentang analitik = 0,65 (hitungan kami dari Tabel 9 Kim, bukan klaim Kim).

## Versi: 6 aspek

77.697 pengamatan dari 6.475 esai (esai skor ekstrem dikeluarkan: 3), 27 penilai, 6 kriteria; 12,0 pengamatan per esai. Konvergen: ya (816 iterasi). Komponen terhubung: 1.

### Ukuran utama

| Ukuran | Penilai (keketatan) | Kriteria (kesulitan) |
|---|---|---|
| Banyak elemen | 27 | 6 |
| Rentang (logit) [galat estimasi, CI 95%] | 4,21 [4,13; 4,30] | 1,17 [1,11; 1,22] |
| Simpangan baku (logit) | 1,03 | 0,40 |
| Indeks pemisahan G | 15,04 | 22,10 |
| Keandalan pemisahan R | 0,996 | 0,998 |
| Khi-kuadrat tetap (db) | 21537 (26) | 2249 (5) |
| Di luar infit/outfit 0,5-1,5 (Kim) | 0 dari 27 | 0 dari 6 |
| Di luar infit/outfit 0,6-1,4 (Wu dkk.) | 0 dari 27 | 0 dari 6 |

Rasio rentang penilai : kriteria = 3,61 [3,44; 3,80] (kebalikannya, kriteria : penilai seperti cara Kim: 0,28). Rasio simpangan baku = 2,54. Hanya penilai dengan ≥ 100 esai (23 penilai): rentang 4,21, rasio 3,61.

Pembanding tanpa model (poin skor): rentang keketatan linear penilai 0,97, rentang kesulitan linear kriteria 0,20, rasio 4,74. Korelasi keketatan MFRM dengan linear: Pearson 1,000, peringkat 0,999; korelasi kesulitan MFRM dengan linear: 0,925. Selisih skor harapan penilai terhadap penilai berketatan rata-rata, pada esai berkemampuan rata-rata (positif = skor lebih rendah): -0,47 sampai 0,53 poin (rentang 1,01 poin). Pengamatan paling sedikit pada satu kategori skor satu kriteria: 18.

### Keketatan tiap penilai (urut dari paling longgar)

| Penilai | Esai | Keketatan logit [CI 95%] | Infit | Outfit | Selisih skor (poin) | Linear (poin) |
|---|---|---|---|---|---|---|
| rater_19 | 780 | -1,84 [-1,90; -1,78] | 1,09 | 1,12 | -0,47 | -0,43 |
| rater_17 | 79 | -1,54 [-1,72; -1,35] | 0,99 | 1,00 | -0,39 | -0,35 |
| rater_15 | 781 | -1,37 [-1,43; -1,31] | 1,15 | 1,14 | -0,34 | -0,32 |
| rater_20 | 60 | -1,18 [-1,40; -0,97] | 0,87 | 0,87 | -0,29 | -0,27 |
| rater_16 | 338 | -0,87 [-0,96; -0,79] | 0,94 | 0,94 | -0,21 | -0,20 |
| rater_5 | 130 | -0,71 [-0,85; -0,57] | 1,08 | 1,07 | -0,17 | -0,16 |
| rater_13 | 689 | -0,71 [-0,77; -0,65] | 0,88 | 0,88 | -0,17 | -0,16 |
| rater_25 | 23 | -0,68 [-1,02; -0,34] | 0,91 | 0,92 | -0,16 | -0,16 |
| rater_12 | 914 | -0,59 [-0,64; -0,53] | 0,95 | 0,93 | -0,14 | -0,14 |
| rater_23 | 569 | -0,46 [-0,53; -0,39] | 0,91 | 0,91 | -0,11 | -0,10 |
| rater_4 | 444 | -0,27 [-0,35; -0,19] | 1,03 | 1,03 | -0,06 | -0,06 |
| rater_14 | 24 | -0,19 [-0,53; 0,16] | 0,76 | 0,71 | -0,04 | -0,05 |
| rater_27 | 330 | -0,15 [-0,24; -0,06] | 1,15 | 1,19 | -0,03 | -0,04 |
| rater_22 | 194 | -0,10 [-0,22; 0,02] | 0,98 | 0,98 | -0,02 | -0,02 |
| rater_24 | 177 | -0,05 [-0,17; 0,08] | 0,99 | 1,00 | -0,01 | -0,01 |
| rater_18 | 825 | 0,02 [-0,03; 0,08] | 0,81 | 0,80 | 0,01 | 0,01 |
| rater_2 | 1.095 | 0,08 [0,03; 0,13] | 1,02 | 1,03 | 0,02 | 0,02 |
| rater_6 | 839 | 0,38 [0,32; 0,44] | 1,15 | 1,17 | 0,08 | 0,09 |
| rater_3 | 228 | 0,41 [0,30; 0,52] | 1,00 | 1,00 | 0,09 | 0,10 |
| rater_1 | 585 | 0,59 [0,53; 0,66] | 0,97 | 0,96 | 0,13 | 0,14 |
| rater_10 | 556 | 0,66 [0,59; 0,73] | 1,14 | 1,13 | 0,14 | 0,15 |
| rater_26 | 164 | 0,67 [0,54; 0,81] | 0,95 | 0,94 | 0,14 | 0,16 |
| rater_7 | 783 | 0,86 [0,80; 0,92] | 1,20 | 1,24 | 0,18 | 0,20 |
| rater_21 | 289 | 1,27 [1,17; 1,36] | 0,85 | 0,85 | 0,27 | 0,29 |
| rater_8 | 303 | 1,31 [1,22; 1,41] | 1,05 | 1,06 | 0,28 | 0,30 |
| rater_11 | 1.132 | 2,08 [2,03; 2,13] | 0,87 | 0,87 | 0,46 | 0,47 |
| rater_9 | 619 | 2,37 [2,30; 2,44] | 0,92 | 0,93 | 0,53 | 0,54 |

### Kesulitan tiap kriteria (urut dari paling mudah)

| Kriteria | Kesulitan logit | SE | Infit | Outfit | Linear (poin) |
|---|---|---|---|---|---|
| Vocabulary | -0,76 | 0,019 | 0,99 | 0,99 | -0,130 |
| Phraseology | -0,09 | 0,018 | 0,93 | 0,92 | -0,015 |
| Cohesion | 0,11 | 0,018 | 1,04 | 1,05 | -0,023 |
| Grammar | 0,15 | 0,018 | 1,04 | 1,05 | 0,074 |
| Conventions | 0,18 | 0,018 | 1,06 | 1,07 | 0,024 |
| Syntax | 0,40 | 0,018 | 0,95 | 0,95 | 0,071 |

### Bias penilai x kriteria (Wu dkk.: tanda bila |t| > 2,0 dan |bias| > 0,5 logit)

150 sel dengan ≥ 30 pengamatan. Bertanda: 49 sel; tetap bertanda setelah Benjamini-Hochberg (q < 0,05): 49 sel. Sel dengan q < 0,05 tanpa syarat ukuran: 93. |Bias| terbesar: 1,50 logit.

| Kriteria | lebih ketat | lebih longgar |
|---|---|---|
| Cohesion | 5 | 3 |
| Syntax | 4 | 3 |
| Vocabulary | 5 | 6 |
| Phraseology | 4 | 3 |
| Grammar | 3 | 4 |
| Conventions | 3 | 6 |

| Penilai | Kriteria | Pengamatan | Bias (logit) | t | Arah |
|---|---|---|---|---|---|
| rater_15 | Cohesion | 781 | -0,92 | -12,5 | lebih longgar |
| rater_11 | Cohesion | 1132 | 0,78 | 12,5 | lebih ketat |
| rater_27 | Vocabulary | 330 | 1,50 | 12,4 | lebih ketat |
| rater_15 | Vocabulary | 781 | 0,97 | 12,2 | lebih ketat |
| rater_12 | Grammar | 914 | 0,75 | 11,5 | lebih ketat |
| rater_7 | Phraseology | 783 | 0,83 | 11,3 | lebih ketat |
| rater_15 | Grammar | 781 | 0,77 | 10,9 | lebih ketat |
| rater_19 | Grammar | 780 | -0,77 | -10,8 | lebih longgar |
| rater_2 | Conventions | 1095 | 0,63 | 10,4 | lebih ketat |
| rater_12 | Phraseology | 914 | -0,69 | -10,2 | lebih longgar |

### Pemeriksaan estimasi (pemulihan parameter)

Data dibangkitkan dari parameter hasil estimasi lalu diestimasi ulang (3 ulangan): korelasi keketatan 0,992; kemiringan keketatan estimasi terhadap sebenarnya 1,106; kemiringan kesulitan 1,117; rasio rentang hasil ulang dibanding rasio rentang sebenarnya 0,973. Perkiraan teori pembesaran JMLE L/(L-1) = 1,091 dengan L = 12,0 pengamatan per esai.

## Versi: 6 aspek + Overall

90.647 pengamatan dari 6.475 esai (esai skor ekstrem dikeluarkan: 3), 27 penilai, 7 kriteria; 14,0 pengamatan per esai. Konvergen: ya (826 iterasi). Komponen terhubung: 1.

### Ukuran utama

| Ukuran | Penilai (keketatan) | Kriteria (kesulitan) |
|---|---|---|
| Banyak elemen | 27 | 7 |
| Rentang (logit) [galat estimasi, CI 95%] | 4,60 [4,52; 4,69] | 1,21 [1,16; 1,26] |
| Simpangan baku (logit) | 1,10 | 0,38 |
| Indeks pemisahan G | 16,94 | 20,44 |
| Keandalan pemisahan R | 0,997 | 0,998 |
| Khi-kuadrat tetap (db) | 27305 (26) | 2319 (6) |
| Di luar infit/outfit 0,5-1,5 (Kim) | 0 dari 27 | 0 dari 7 |
| Di luar infit/outfit 0,6-1,4 (Wu dkk.) | 0 dari 27 | 0 dari 7 |

Rasio rentang penilai : kriteria = 3,80 [3,63; 3,99] (kebalikannya, kriteria : penilai seperti cara Kim: 0,26). Rasio simpangan baku = 2,87. Hanya penilai dengan ≥ 100 esai (23 penilai): rentang 4,60, rasio 3,80.

Pembanding tanpa model (poin skor): rentang keketatan linear penilai 1,00, rentang kesulitan linear kriteria 0,20, rasio 4,91. Korelasi keketatan MFRM dengan linear: Pearson 1,000, peringkat 0,999; korelasi kesulitan MFRM dengan linear: 0,921. Selisih skor harapan penilai terhadap penilai berketatan rata-rata, pada esai berkemampuan rata-rata (positif = skor lebih rendah): -0,49 sampai 0,55 poin (rentang 1,04 poin). Pengamatan paling sedikit pada satu kategori skor satu kriteria: 18.

### Keketatan tiap penilai (urut dari paling longgar)

| Penilai | Esai | Keketatan logit [CI 95%] | Infit | Outfit | Selisih skor (poin) | Linear (poin) |
|---|---|---|---|---|---|---|
| rater_19 | 780 | -1,99 [-2,05; -1,94] | 1,08 | 1,10 | -0,49 | -0,44 |
| rater_17 | 79 | -1,61 [-1,79; -1,43] | 0,99 | 0,99 | -0,39 | -0,35 |
| rater_15 | 781 | -1,51 [-1,56; -1,45] | 1,12 | 1,11 | -0,36 | -0,33 |
| rater_20 | 60 | -1,26 [-1,46; -1,05] | 0,87 | 0,86 | -0,29 | -0,27 |
| rater_16 | 338 | -0,92 [-1,01; -0,84] | 0,94 | 0,94 | -0,21 | -0,20 |
| rater_13 | 689 | -0,78 [-0,84; -0,72] | 0,88 | 0,87 | -0,17 | -0,17 |
| rater_5 | 130 | -0,71 [-0,85; -0,57] | 1,04 | 1,02 | -0,16 | -0,15 |
| rater_25 | 23 | -0,64 [-0,96; -0,32] | 0,93 | 0,94 | -0,14 | -0,14 |
| rater_12 | 914 | -0,60 [-0,65; -0,55] | 0,96 | 0,94 | -0,13 | -0,13 |
| rater_23 | 569 | -0,46 [-0,53; -0,39] | 0,91 | 0,91 | -0,10 | -0,10 |
| rater_4 | 444 | -0,35 [-0,43; -0,28] | 1,04 | 1,03 | -0,07 | -0,08 |
| rater_14 | 24 | -0,24 [-0,57; 0,09] | 0,79 | 0,73 | -0,05 | -0,06 |
| rater_22 | 194 | -0,16 [-0,27; -0,05] | 0,99 | 0,99 | -0,03 | -0,03 |
| rater_27 | 330 | -0,14 [-0,23; -0,06] | 1,14 | 1,18 | -0,03 | -0,03 |
| rater_24 | 177 | -0,11 [-0,22; 0,01] | 1,01 | 1,02 | -0,02 | -0,03 |
| rater_18 | 825 | -0,03 [-0,08; 0,03] | 0,82 | 0,79 | -0,01 | -0,01 |
| rater_2 | 1.095 | 0,10 [0,06; 0,15] | 1,02 | 1,03 | 0,02 | 0,02 |
| rater_3 | 228 | 0,42 [0,32; 0,53] | 0,99 | 0,98 | 0,08 | 0,09 |
| rater_6 | 839 | 0,46 [0,40; 0,51] | 1,13 | 1,14 | 0,09 | 0,10 |
| rater_1 | 585 | 0,66 [0,60; 0,73] | 0,97 | 0,96 | 0,13 | 0,15 |
| rater_26 | 164 | 0,69 [0,56; 0,82] | 0,94 | 0,93 | 0,13 | 0,15 |
| rater_10 | 556 | 0,76 [0,69; 0,82] | 1,13 | 1,13 | 0,14 | 0,17 |
| rater_7 | 783 | 0,88 [0,82; 0,93] | 1,20 | 1,23 | 0,17 | 0,20 |
| rater_21 | 289 | 1,34 [1,24; 1,43] | 0,86 | 0,85 | 0,26 | 0,29 |
| rater_8 | 303 | 1,46 [1,37; 1,56] | 1,04 | 1,06 | 0,28 | 0,32 |
| rater_11 | 1.132 | 2,14 [2,09; 2,19] | 0,91 | 0,91 | 0,44 | 0,46 |
| rater_9 | 619 | 2,61 [2,54; 2,67] | 0,95 | 0,96 | 0,55 | 0,56 |

### Kesulitan tiap kriteria (urut dari paling mudah)

| Kriteria | Kesulitan logit | SE | Infit | Outfit | Linear (poin) |
|---|---|---|---|---|---|
| Vocabulary | -0,80 | 0,020 | 1,04 | 1,05 | -0,130 |
| Phraseology | -0,10 | 0,019 | 0,98 | 0,98 | -0,015 |
| Overall | 0,08 | 0,019 | 0,66 | 0,60 | -0,003 |
| Cohesion | 0,10 | 0,018 | 1,09 | 1,12 | -0,023 |
| Grammar | 0,14 | 0,018 | 1,11 | 1,13 | 0,074 |
| Conventions | 0,17 | 0,018 | 1,12 | 1,14 | 0,024 |
| Syntax | 0,41 | 0,019 | 0,99 | 1,00 | 0,071 |

### Bias penilai x kriteria (Wu dkk.: tanda bila |t| > 2,0 dan |bias| > 0,5 logit)

175 sel dengan ≥ 30 pengamatan. Bertanda: 58 sel; tetap bertanda setelah Benjamini-Hochberg (q < 0,05): 58 sel. Sel dengan q < 0,05 tanpa syarat ukuran: 104. |Bias| terbesar: 1,50 logit.

| Kriteria | lebih ketat | lebih longgar |
|---|---|---|
| Cohesion | 5 | 5 |
| Syntax | 4 | 3 |
| Vocabulary | 5 | 6 |
| Phraseology | 4 | 4 |
| Grammar | 5 | 6 |
| Conventions | 4 | 6 |
| Overall | 1 | 0 |

| Penilai | Kriteria | Pengamatan | Bias (logit) | t | Arah |
|---|---|---|---|---|---|
| rater_11 | Cohesion | 1132 | 0,86 | 13,6 | lebih ketat |
| rater_15 | Vocabulary | 781 | 1,06 | 12,8 | lebih ketat |
| rater_15 | Cohesion | 781 | -0,96 | -12,5 | lebih longgar |
| rater_27 | Vocabulary | 330 | 1,50 | 12,1 | lebih ketat |
| rater_7 | Phraseology | 783 | 0,91 | 12,1 | lebih ketat |
| rater_15 | Grammar | 781 | 0,84 | 11,5 | lebih ketat |
| rater_12 | Phraseology | 914 | -0,77 | -11,1 | lebih longgar |
| rater_12 | Grammar | 914 | 0,74 | 11,0 | lebih ketat |
| rater_19 | Grammar | 780 | -0,78 | -10,7 | lebih longgar |
| rater_2 | Conventions | 1095 | 0,66 | 10,6 | lebih ketat |

### Pemeriksaan estimasi (pemulihan parameter)

Data dibangkitkan dari parameter hasil estimasi lalu diestimasi ulang (3 ulangan): korelasi keketatan 0,994; kemiringan keketatan estimasi terhadap sebenarnya 1,092; kemiringan kesulitan 1,191; rasio rentang hasil ulang dibanding rasio rentang sebenarnya 0,895. Perkiraan teori pembesaran JMLE L/(L-1) = 1,077 dengan L = 14,0 pengamatan per esai.

## Sensitivitas menurut himpunan baris

| Himpunan | Versi | Baris | Rentang penilai (logit) | Rentang kriteria (logit) | Rasio rentang | Rasio SD | Rentang selisih skor penilai (poin) |
|---|---|---|---|---|---|---|---|
| Baris ber-text_id_kaggle (himpunan resmi) (utama) | 6 aspek | 6.478 | 4,21 | 1,17 | 3,61 | 2,54 | 1,01 |
| Baris ber-text_id_kaggle (himpunan resmi) (utama) | 6 aspek + Overall | 6.478 | 4,60 | 1,21 | 3,80 | 2,87 | 1,04 |
| Semua baris | 6 aspek | 8.859 | 2,60 | 0,90 | 2,89 | 2,06 | 0,80 |
| Semua baris | 6 aspek + Overall | 8.859 | 2,87 | 0,94 | 3,07 | 2,34 | 0,83 |
| Baris tanpa text_id_kaggle | 6 aspek | 2.381 | 1,93 | 0,68 | 2,85 | 1,92 | 0,97 |
| Baris tanpa text_id_kaggle | 6 aspek + Overall | 2.381 | 2,02 | 0,70 | 2,87 | 2,16 | 0,95 |

Logit dari dua himpunan yang berbeda tidak sebanding langsung (ambang dan skala diestimasi terpisah di tiap himpunan); rasio dan poin skor sebanding.
