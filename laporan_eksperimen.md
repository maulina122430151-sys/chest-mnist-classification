# Laporan Eksperimen: Klasifikasi Biner ChestMNIST

| Informasi | Detail |
|-----------|--------|
| Nama | Maulina Adelia Putri |
| NIM | 122430151 |
| Program Studi | Teknik Biomedis |
| Mata Kuliah | Kecerdasan Buatan |

## 📋 Ringkasan Proyek

### Tujuan
Membangun model klasifikasi biner untuk membedakan antara dua kondisi medis pada dataset ChestMNIST:
- **Class 0**: Cardiomegaly (Pembesaran Jantung)
- **Class 1**: Pneumothorax (Kolaps Paru-paru)

**Target Akurasi**: ≥ 90% validation accuracy

## 📊 Hasil Eksperimen dan Analisis

### Arsitektur Model
Dalam eksperimen ini, kami menggunakan arsitektur ResNet dengan modifikasi berikut:
- Input Layer: 1 channel (grayscale X-ray)
- 3 ResNet Blocks dengan peningkatan channel (64 → 128 → 256)
- Global Average Pooling
- Fully Connected Layer untuk output biner

### Parameter Training
- Epochs: 20
- Batch Size: 10
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: BCEWithLogitsLoss

### Analisis Hasil Training

#### Metrik Performa
- **Training Accuracy**: ~88%
- **Validation Accuracy**: ~85%
- **Training Loss**: 0.2845
- **Validation Loss**: 0.3156

#### Analisis Kurva Learning
1. **Konvergensi**:
   - Model menunjukkan pembelajaran yang stabil
   - Loss menurun secara konsisten selama training
   - Tidak ada tanda overfitting yang signifikan

2. **Stabilitas Training**:
   - Kurva training dan validation berjalan paralel
   - Gap antara training dan validation accuracy relatif kecil
   - Menunjukkan generalisasi yang baik

3. **Pola Validasi**:
   - Validation accuracy meningkat secara konsisten
   - Fluktuasi validation loss minimal
   - Menunjukkan model yang robust

## 🎯 Kesimpulan

### Pencapaian Target
- Model mencapai performa yang cukup baik meskipun belum mencapai target 90%
- Hasil menunjukkan potensi untuk peningkatan lebih lanjut

### Rekomendasi Peningkatan
1. **Arsitektur Model**:
   - Menambah jumlah layer atau filter
   - Implementasi skip connections tambahan

2. **Hyperparameter Tuning**:
   - Eksperimen dengan learning rate yang lebih rendah
   - Meningkatkan jumlah epochs
   - Implementasi learning rate scheduling

3. **Data Preprocessing**:
   - Augmentasi data tambahan
   - Implementasi teknik balancing dataset

### Kesimpulan Akhir
Model ResNet yang diimplementasikan menunjukkan performa yang menjanjikan dalam klasifikasi Cardiomegaly vs Pneumothorax. Meskipun belum mencapai target 90%, hasil eksperimen memberikan insight berharga untuk pengembangan lebih lanjut.