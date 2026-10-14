# Replikasi Parsial Spark untuk Peramalan Beban Listrik

Repositori ini menyertai tugas Eksplorasi Mandiri Mahadata milik **Jana Rohman Wasiso (123140046)**.

## Artikel utama

H. Ali El-Sayed Ali, M. H. Alham, dan D. K. Ibrahim (2024), *Big data resolving using Apache Spark for load forecasting and demand response in smart grid: a case study of Low Carbon London Project*, Journal of Big Data, 11, 59. https://doi.org/10.1186/s40537-024-00909-6

Eksperimen ini adalah **replikasi parsial**: data smart-meter Low Carbon London diproses menggunakan Spark SQL dan Spark MLlib dalam mode lokal. Artikel asli menggunakan seluruh data (lebih dari 167 juta pembacaan), sedangkan implementasi ini mengunduh satu berkas CSV terkompresi dari arsip resmi dan memakai fitur kalender saja. Karena itu, angka hasil tidak boleh diklaim sebagai replikasi penuh.

## Struktur

```text
src/download_lcl_sample.py  mengambil satu CSV dari ZIP besar dengan HTTP Range
src/run_experiment.py       pembersihan, agregasi, pelatihan, evaluasi, grafik
data/raw/                   CSV yang diunduh (tidak di-commit)
results/                    metrik CSV, prediksi CSV, grafik PNG, metadata JSON
```

## Prasyarat

- Python 3.10 atau lebih baru
- Java 17 (atau Java 11)
- `pip install -r requirements.txt`

## Cara menjalankan

Pada PowerShell dari folder repositori:

```powershell
python src/download_lcl_sample.py
python src/run_experiment.py
```

Skrip kedua menjalankan Spark secara lokal (`local[*]`). Hasil utamanya adalah:

- `results/metrics.csv`
- `results/predictions.csv`
- `results/actual_vs_prediction.png`
- `results/metadata.json`

## Reprodusibilitas dan batasan

Random seed model ditetapkan ke `42`. Pemisahan data dilakukan berdasarkan waktu: 80% timestamp awal untuk latih dan 20% timestamp akhir untuk uji. Ini lebih sesuai untuk peramalan daripada pemisahan acak, tetapi berbeda dari konfigurasi penuh artikel. Dataset sumber berasal dari UK Power Networks / London Datastore, bukan UCI Online Retail.

