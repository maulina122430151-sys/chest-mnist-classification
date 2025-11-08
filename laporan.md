# Laporan Eksperimen: Klasifikasi Biner ChestMNIST

**Nama:** Maulina Adelia Putri  
**NIM:** 122430151  
**Program Studi:** Teknik Biomedis  
**Mata Kuliah:** Kecerdasan Buatan

## 📋 Ringkasan Proyek

**Tujuan:** Klasifikasi citra X-ray dada untuk membedakan:
- Class 0: Cardiomegaly (Pembesaran Jantung)
- Class 1: Pneumothorax (Kolaps Paru-paru)

## � Model dan Training

**Model:** ResNet
- Input: Citra X-ray (1 channel)
- 3 ResNet Blocks (64 → 128 → 256 channels)
- Output: Klasifikasi biner

**Parameter Training:**
- Epochs: 20
- Batch Size: 10
- Learning Rate: 0.001
- Optimizer: Adam
- Loss: BCEWithLogitsLoss

## 📊 Hasil dan Analisis

**Performa Model:**
- Training Accuracy: 88.45%
- Validation Accuracy: 85.32%
- Training Loss: 0.2845
- Validation Loss: 0.3156

**Analisis:**
- Model berhasil belajar pola dari data training
- Kurva loss menurun stabil
- Tidak terjadi overfitting signifikan (gap antara training dan validation sekitar 3%)

## ✍️ Kesimpulan
Model ResNet menunjukkan kemampuan yang baik dalam membedakan antara kasus Cardiomegaly dan Pneumothorax pada citra X-ray dada.