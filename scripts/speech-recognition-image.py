import os
import numpy as np
import csv
import pandas as pd
from PIL import Image
from zipfile import ZIP_LZMA as lzma
from zipfile import ZipFile
import librosa
import librosa.display as disp
from python_speech_features import mfcc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
if os.path.exists('wavs'):
    pass
elif os.path.exists('wavs.zip'):
    with ZipFile('wavs.zip', 'r') as zipObj:
        zipObj.extractall()
        zipObj.close()
else:
    raise Exception('wavs folder or zip no found!!')
if os.path.exists('mfccpic'):
    pass
elif os.path.exists('mfccpic.zip'):
    with ZipFile('mfccpic.zip', 'r') as zipObj:
        zipObj.extractall()
        zipObj.close()
else:
    os.mkdir('mfccpic')
    for i in range(10):
        os.mkdir(f'mfccpic/{i}')
        path = f'wavs/{i}'
        fileList = os.listdir(path)
        for j in fileList:
            wav, fs = librosa.load(f'wavs/{i}/{j}')
            m = mfcc(wav, fs, nfft=1024, winfunc=np.hamming)
            fig = plt.imshow(m, aspect='auto', cmap='plasma')
            plt.axis('off')
            plt.savefig(f'mfccpic/{i}/{j}.png',
                        bbox_inches='tight', pad_inches=0)
            plt.clf()
    zipObj = ZipFile('mfccpic.zip', 'w')
    for i in range(10):
        path = f'mfccpic/{i}'
        fileList = os.listdir(path)
        for j in fileList:
            zipObj.write(f'{path}/{j}')
    zipObj.close()
if os.path.exists('imdata.csv'):
    data = pd.read_csv('imdata.csv')

elif os.path.exists('imdata.zip'):
    with ZipFile('imdata.zip', 'r') as zipObj:
        zipObj.extractall()
    data = pd.read_csv('imdata.csv')
else:
    df = pd.DataFrame()
    for i in range(10):
        path = f'mfccpic/{i}'
        fileList = os.listdir(path)
        for j in fileList:
            im = Image.open(f'{path}/{j}').resize((50, 50)).convert('RGB')
            arr = np.array(im).flatten()/255
            arr = np.append(arr, i)
            s = pd.Series(arr)
            df = df.append(s, ignore_index=True)
    df.rename(columns={2500: 'label'}, inplace=True)
    df.to_csv('imdata.csv', index=False)
    zipObj = ZipFile('imdata.zip', 'w', compression=lzma)
    zipObj.write('imdata.csv')
    zipObj.close()
    data = pd.read_csv('imdata.csv')
X = data.iloc[:, :-1].to_numpy()
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
model.fit(X_train, y_train, epochs=20, batch_size=100)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('accuracy: ', test_acc, '\nloss: ', test_loss)
