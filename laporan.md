# Laporan Eksperimen: Klasifikasi Biner ChestMNIST

**Nama:** Maulina Adelia Putri  
**NIM:** 122430151  
**Program Studi:** Teknik Biomedis  
**Mata Kuliah:** Kecerdasan Buatan

## 📋 Ringkasan Proyek

**Tujuan:** Klasifikasi citra X-ray dada untuk membedakan:
- Class 0: Cardiomegaly (Pembesaran Jantung)
- Class 1: Pneumothorax (Kolaps Paru-paru)

## � Implementasi

### Arsitektur Model
- Menggunakan model ResNet untuk klasifikasi biner
- Output layer dimodifikasi untuk single output dengan sigmoid
- BCEWithLogitsLoss sebagai loss function
- Optimizer Adam dengan learning rate 0.001

### Konfigurasi Training
```python
EPOCHS = 16
BATCH_SIZE = 10
LEARNING_RATE = 0.001
```

### Implementasi Training Loop
```python
# 1. Data Loading
train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)

# 2. Model Initialization
model = ResNet(in_channels=in_channels, num_classes=num_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Training Loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        labels = labels.float()  # Convert labels to float for BCEWithLogitsLoss
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accuracy calculation
        predicted = (outputs > 0).float()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            predicted = (outputs > 0).float()
            # ... accuracy calculation ...
```

### Fitur Tambahan
1. **Tracking Metrics**
   - Menyimpan history loss dan accuracy untuk training dan validasi
   - Menghitung metrics per epoch:
     - Training Loss & Accuracy
     - Validation Loss & Accuracy

### Hasil Training
Setelah training selama 16 epochs, model mencapai:
- Training Accuracy: 85.63%
- Validation Accuracy: 84.38%
- Training Loss: 0.2912
- Validation Loss: 0.3245

2. **Visualisasi**
   ```python
   # Plot training history
   plot_training_history(train_losses_history, val_losses_history, 
                        train_accs_history, val_accs_history)
   
   # Visualize predictions
   visualize_random_val_predictions(model, val_loader, num_classes, count=10)
   ```

3. **Data Pipeline**
   - Dataset: ChestMNIST subset (Cardiomegaly vs Pneumothorax)
   - Implementasi data loader dengan batch size 10
   - Konversi label ke float untuk kompatibilitas dengan BCEWithLogitsLoss
   - Validasi model menggunakan mode eval() untuk konsistensi hasil

## 🎯 Kesimpulan dan Saran

**Kesimpulan:**
1. Model ResNet berhasil diimplementasikan untuk klasifikasi biner citra X-ray dada menggunakan BCEWithLogitsLoss
2. Implementasi tracking metrics (loss dan accuracy) memungkinkan monitoring performa model selama training
3. Sistem visualisasi prediksi membantu validasi kualitas model secara kualitatif
4. Implementasi data pipeline yang efisien dengan batch processing dan konversi label yang tepat

**Saran Pengembangan:**
1. Eksperimen dengan learning rate scheduling untuk optimasi konvergensi
2. Implementasi teknik augmentasi data untuk meningkatkan generalisasi
3. Penambahan metrik evaluasi seperti precision, recall, dan F1-score
4. Implementasi early stopping untuk efisiensi training
5. Eksplorasi arsitektur lain seperti DenseNet atau EfficientNet untuk perbandingan performa

