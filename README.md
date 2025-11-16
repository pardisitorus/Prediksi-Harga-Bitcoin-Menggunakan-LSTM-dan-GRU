# LSTM Bitcoin Price Prediction

## Deskripsi Proyek
Proyek ini mengimplementasikan model LSTM (Long Short-Term Memory) untuk memprediksi harga Bitcoin (BTC-USD) berdasarkan data historis. Model menggunakan arsitektur 2 layer LSTM dengan 50 neuron per layer, lookback period 60 timestep, dan dilatih selama 1 epoch.

## Dataset
- **Nama file**: `BTC-USD.csv`
- **Sumber**: Data historis harga Bitcoin USD
- **Download**: File CSV harus didownload terlebih dahulu dari repository GitHub dan disimpan di working directory

## Requirements
Library yang diperlukan:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
```

## Langkah-Langkah Implementasi

### Langkah 1: Import Library
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# Set seed untuk reproducibility
tf.random.set_seed(42)
np.random.seed(42)
```

### Langkah 2: Load Dataset dan Preprocessing Data
```python
# Load dataset
df = pd.read_csv('BTC-USD.csv')

# Ubah kolom Date menjadi datetime dan jadikan index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Hanya gunakan kolom Close
data = df[['Close']].copy()

print("5 data teratas:")
print(data.head())
```

### Langkah 3: Scaling Data ke Range [0, 1]
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(data.values)

print("\nShape data setelah scaling:", scaled_values.shape)
```

### Langkah 4: Split Data Train (80%) dan Test (20%)
```python
training_data_len = int(len(scaled_values) * 0.8)

train_data = scaled_values[:training_data_len]
lookback = 60
test_data = scaled_values[training_data_len - lookback:]

print("Jumlah data train:", len(train_data))
print("Jumlah data test (dengan tambahan lookback):", len(test_data))
```

### Langkah 5: Fungsi Pembuat Sequence (X, y) dengan Lookback = 60
```python
def create_sequences(dataset, lookback):
    """
    dataset : array 2D (N, 1)
    lookback: banyak timestep yang digunakan (misal 60 hari)
    return  : X (samples, lookback), y (samples,)
    """
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i - lookback:i, 0])  # 60 hari sebelumnya
        y.append(dataset[i, 0])               # hari ke-61 (target)
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, lookback)
X_test, y_test = create_sequences(test_data, lookback)

print("\nShape X_train sebelum reshape:", X_train.shape)
print("Shape X_test sebelum reshape:", X_test.shape)
```

### Langkah 6: Reshape Data ke Format 3D untuk LSTM
```python
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("Shape X_train setelah reshape:", X_train.shape)
print("Shape X_test setelah reshape:", X_test.shape)
```

### Langkah 7: Membangun Model LSTM
```python
def build_lstm_model(neurons=50, lookback=60):
    model = Sequential()
    # LSTM layer pertama
    model.add(LSTM(neurons,
                   return_sequences=True,
                   input_shape=(lookback, 1)))
    # LSTM layer kedua
    model.add(LSTM(neurons, return_sequences=False))
    # Output layer (1 neuron)
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = build_lstm_model(neurons=50, lookback=lookback)
lstm_model.summary()
```

### Langkah 8: Training Model LSTM
```python
epochs = 1

history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=32,
    verbose=1
)
```

### Langkah 9: Prediksi pada Data Test dan Inverse Scaling
```python
pred_lstm_scaled = lstm_model.predict(X_test)

# Kembalikan ke skala harga asli
pred_lstm = scaler.inverse_transform(pred_lstm_scaled)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
```

### Langkah 10: Evaluasi Model dengan Metrik RMSE, MAE, MAPE
```python
rmse_lstm = np.sqrt(mean_squared_error(y_test_unscaled, pred_lstm))
mae_lstm = mean_absolute_error(y_test_unscaled, pred_lstm)
mape_lstm = np.mean(np.abs((y_test_unscaled - pred_lstm) / y_test_unscaled)) * 100

print("\n=== EVALUASI MODEL LSTM ===")
print("RMSE LSTM :", rmse_lstm)
print("MAE LSTM :", mae_lstm)
print("MAPE LSTM : {:.2f}%".format(mape_lstm))
```

### Langkah 11: Persiapan Data untuk Visualisasi
```python
train = data.iloc[:training_data_len]
valid = data.iloc[training_data_len:].copy()

# Samakan panjang valid dengan panjang prediksi
valid = valid.iloc[-len(pred_lstm):].copy()
valid['Pred_LSTM'] = pred_lstm

print("\nPanjang valid:", len(valid))
print("Panjang pred_lstm:", len(pred_lstm))
```

### Langkah 12: Plot Hasil Prediksi vs Data Aktual
```python
plt.figure(figsize=(12, 6))
plt.plot(train['Close'], label='Train (Close)')
plt.plot(valid['Close'], label='Actual Test (Close)')
plt.plot(valid['Pred_LSTM'], label='Predicted LSTM')
plt.title('Prediksi Harga Bitcoin dengan LSTM')
plt.xlabel('Tanggal')
plt.ylabel('Harga Close')
plt.legend()
plt.grid(True)
plt.show()
```

## Analisis Hasil

### Arsitektur Model
- **Layer LSTM**: 2 layer
- **Neuron per Layer**: 50
- **Lookback Period**: 60 timestep (hari)
- **Epoch Training**: 1
- **Batch Size**: 32

### Metrik Evaluasi
Berdasarkan output yang dihasilkan:
- **RMSE**: 2793.75
- **MAE**: 1955.80
- **MAPE**: 6.12%

### Interpretasi Hasil
1. **MAPE 6.12%** menunjukkan model memiliki akurasi yang cukup baik dengan error rata-rata sekitar 6% dari nilai aktual
2. **RMSE dan MAE** yang tinggi mencerminkan volatilitas tinggi harga Bitcoin
3. Model mampu menangkap pola umum pergerakan harga meskipun dengan training yang singkat (1 epoch)

## Kesimpulan

1. **Implementasi Berhasil**: Model LSTM berhasil dibangun dan dilatih untuk memprediksi harga Bitcoin
2. **Akurasi Memadai**: Dengan MAPE 6.12%, model menunjukkan performa yang cukup baik untuk prediksi time series finansial
3. **Efisiensi Training**: Meski hanya 1 epoch, model sudah mampu belajar pola data
4. **Potensi Improvement**:
   - Training lebih banyak epoch
   - Tuning hyperparameter
   - Penambahan feature engineering
   - Eksperimen dengan arsitektur yang lebih kompleks

5. **Aplikabilitas**: Model ini dapat digunakan sebagai dasar untuk sistem prediksi harga cryptocurrency dengan berbagai penyempurnaan lebih lanjut.

Proyek ini memberikan fondasi yang solid untuk pengembangan lebih lanjut dalam prediksi time series finansial menggunakan deep learning.
