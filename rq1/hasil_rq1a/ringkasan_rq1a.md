# RQ1a: seberapa setuju dua penilai

Himpunan utama: Baris ber-text_id_kaggle (himpunan resmi).

## Kesepakatan dan batas (CI 95%, bootstrap per pasangan penilai)

| Kriteria | n | QWK | Pearson | ICC(1,1) | Persis sama % | Selisih ≥ 2 % | κ_HL | κ_max |
|---|---|---|---|---|---|---|---|---|
| Overall | 6478 | 0,591 [0,568; 0,611] | 0,592 | 0,591 | 59,2 | 0,3 | 0,663 [0,641; 0,680] | 0,862 [0,851; 0,871] |
| Cohesion | 6477 | 0,497 [0,460; 0,531] | 0,500 | 0,496 | 50,1 | 2,6 | 0,574 [0,538; 0,605] | 0,815 [0,794; 0,832] |
| Syntax | 6477 | 0,511 [0,477; 0,540] | 0,513 | 0,510 | 52,9 | 1,8 | 0,587 [0,554; 0,615] | 0,822 [0,803; 0,837] |
| Vocabulary | 6478 | 0,484 [0,457; 0,507] | 0,485 | 0,484 | 56,6 | 1,0 | 0,562 [0,535; 0,583] | 0,808 [0,792; 0,820] |
| Phraseology | 6478 | 0,511 [0,483; 0,535] | 0,512 | 0,510 | 51,8 | 2,0 | 0,587 [0,559; 0,610] | 0,822 [0,806; 0,835] |
| Grammar | 6478 | 0,532 [0,498; 0,560] | 0,534 | 0,531 | 50,3 | 2,5 | 0,607 [0,574; 0,633] | 0,833 [0,814; 0,847] |
| Conventions | 6477 | 0,519 [0,480; 0,551] | 0,519 | 0,518 | 51,5 | 2,5 | 0,595 [0,556; 0,626] | 0,826 [0,804; 0,843] |

## Aspek dibanding skor keseluruhan dan jarak ke batas

| Aspek | QWK aspek − QWK Overall [CI] | κ_HL − QWK [CI] | κ_max − QWK [CI] |
|---|---|---|---|
| Cohesion | -0,094 [-0,123; -0,067] | 0,076 [0,071; 0,078] | 0,317 [0,301; 0,333] |
| Syntax | -0,080 [-0,105; -0,056] | 0,077 [0,073; 0,078] | 0,311 [0,297; 0,326] |
| Vocabulary | -0,107 [-0,128; -0,086] | 0,078 [0,076; 0,078] | 0,323 [0,313; 0,335] |
| Phraseology | -0,080 [-0,105; -0,056] | 0,076 [0,074; 0,077] | 0,311 [0,300; 0,323] |
| Grammar | -0,059 [-0,089; -0,031] | 0,075 [0,072; 0,077] | 0,301 [0,287; 0,316] |
| Conventions | -0,073 [-0,106; -0,043] | 0,076 [0,074; 0,078] | 0,308 [0,292; 0,325] |

## Selisih QWK antar aspek (CI berpasangan, tanpa koreksi uji berganda)

| Aspek A | Aspek B | QWK A − QWK B [CI] | CI tidak memuat 0 |
|---|---|---|---|
| Vocabulary | Grammar | -0,048 [-0,078; -0,019] | ya |
| Cohesion | Grammar | -0,035 [-0,068; 0,001] | tidak |
| Vocabulary | Conventions | -0,034 [-0,065; -0,004] | ya |
| Vocabulary | Phraseology | -0,026 [-0,049; -0,003] | ya |
| Syntax | Grammar | -0,021 [-0,061; 0,018] | tidak |
| Phraseology | Grammar | -0,021 [-0,049; 0,006] | tidak |
| Cohesion | Conventions | -0,021 [-0,052; 0,011] | tidak |
| Cohesion | Phraseology | -0,014 [-0,045; 0,016] | tidak |
| Cohesion | Syntax | -0,013 [-0,052; 0,019] | tidak |
| Syntax | Conventions | -0,008 [-0,047; 0,036] | tidak |
| Phraseology | Conventions | -0,008 [-0,040; 0,027] | tidak |
| Syntax | Phraseology | -0,000 [-0,038; 0,036] | tidak |
| Cohesion | Vocabulary | 0,013 [-0,019; 0,044] | tidak |
| Grammar | Conventions | 0,014 [-0,019; 0,047] | tidak |
| Syntax | Vocabulary | 0,026 [-0,003; 0,056] | tidak |

## Tabel silang skor keseluruhan (baris = penilai slot 1, kolom = penilai slot 2)

| Slot 1 \ Slot 2 | 1 | 2 | 3 | 4 | 5 | Jumlah |
|---|---|---|---|---|---|---|
| 1 | 11 | 18 | 1 | 0 | 0 | 30 |
| 2 | 15 | 554 | 637 | 7 | 0 | 1213 |
| 3 | 1 | 501 | 2322 | 548 | 1 | 3373 |
| 4 | 0 | 8 | 755 | 906 | 68 | 1737 |
| 5 | 0 | 0 | 0 | 85 | 40 | 125 |
| Jumlah | 27 | 1081 | 3715 | 1546 | 109 | 6478 |

## Sensitivitas menurut himpunan baris (QWK [CI 95%])

| Himpunan | n | Overall | Cohesion | Syntax | Vocabulary | Phraseology | Grammar | Conventions | κ_HL Overall | κ_max Overall | Persis % | ≥ 2 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baris ber-text_id_kaggle (himpunan resmi) (utama) | 6478 | 0,591 [0,568; 0,611] | 0,50 [0,46; 0,53] | 0,51 [0,48; 0,54] | 0,48 [0,46; 0,51] | 0,51 [0,48; 0,53] | 0,53 [0,50; 0,56] | 0,52 [0,48; 0,55] | 0,663 | 0,862 | 59,2 | 0,3 |
| Semua baris | 8859 | 0,511 [0,486; 0,536] | 0,42 [0,39; 0,46] | 0,43 [0,39; 0,46] | 0,39 [0,36; 0,42] | 0,42 [0,39; 0,45] | 0,46 [0,43; 0,50] | 0,46 [0,42; 0,49] | 0,588 | 0,823 | 53,0 | 2,9 |
| Baris tanpa text_id_kaggle | 2381 | 0,359 [0,301; 0,418] | 0,29 [0,24; 0,34] | 0,27 [0,22; 0,32] | 0,23 [0,18; 0,28] | 0,27 [0,21; 0,32] | 0,33 [0,26; 0,40] | 0,34 [0,27; 0,40] | 0,435 | 0,727 | 36,1 | 10,1 |
