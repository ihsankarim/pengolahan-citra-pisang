# Proyek Klasifikasi Pisang

## Ringkasan Proyek
Proyek machine learning untuk mengklasifikasikan berbagai jenis pisang menggunakan teknik pengenalan gambar.

## Prasyarat
- Python 3.8+
- pip (Manajer paket Python)

## Pengaturan dan Instalasi

1. Instal dependensi
```bash
pip install -r requirements.txt atau pip3 install -r requirements.txt
```

## Struktur Proyek
- `model.py`: Berisi definisi model jaringan saraf (CNN)
- `preprocessing.py`: Pra-pemrosesan gambar dan augmentasi data
- `retrain.py`: Skrip untuk melatih ulang model
- `predict.py`: Skrip untuk melakukan prediksi pada gambar baru

## Menjalankan Proyek
1. Membuat model
```bash
python retrain.py | python3 retrain.py 
```

2. Melakukan prediksi
```bash
python predict.py | python3 predict.py
```

### Catatan Penting
- Pastikan sudah menginstal semua dependensi sebelum menjalankan skrip
- Gunakan `python` atau `python3` sesuai dengan konfigurasi Python di komputer 