import os
import numpy as np
import csv
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
if os.path.exists('data.csv'):
    data = pd.read_csv('data.csv')
else:
    cols = ['filename', 'chroma_stft', 'rms', 'spectral_centroid',
            'spectral_bandwidth', 'rolloff', 'zero_crossing_rate']
    for i in range(20):
        cols.append(f'mfcc_{i}')
    cols.append('label')
    f = open('data.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(cols)
    f.close()
    for i in range(10):
        path = f'/content/drive/My Drive/wavs/{i}'
        fileList = os.listdir(path)
        for j in fileList:
            y, sr = librosa.load(f'{path}/{j}', mono=True)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            row = [f'{j}', np.mean(chroma_stft), np.mean(rms), np.mean(
                spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
            for val in range(20):
                row.append(np.mean(mfcc[val]))
            row.append(i)
            f = open('data.csv', 'a', newline='')
            writer = csv.writer(f)
            writer.writerow(row)
            f.close()
data = data.drop(['filename'], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(data.iloc[:, :-1])
y = data.iloc[:, -1].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu',
                          input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=100)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('accuracy: ', test_acc, '\nloss: ', test_loss)
