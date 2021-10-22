import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy

import librosa
from librosa import display
from IPython.display import Audio 

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

import os
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import model_selection
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import mean_absolute_error
import os
import librosa
import json
import codecs
import python_speech_features
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from scipy.signal.windows import hann
import seaborn as sns
from scipy import signal

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from keras.models import model_from_json
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint 
from datetime import datetime
import tensorflow as tf

import keras
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import keras.backend as K
from sklearn.preprocessing import minmax_scale
import pickle
from numpy import savetxt
from numpy import asarray

def audio_features_extraction(sample, sample_rate, n_fft):
    
    """ 
        sample - audio time series
        sample_rate - sampling rate of sample
        n_fft = frame size
    """
    
    # librosa.feature.mfcc - вычилсяет коэффициенты MFCCs.
    # MFCCs трансформируют значение сигнала в кепстр – один из видов гомоморфной обработки сигналов, 
    # функция обратного преобразования Фурье от логарифма спектра мощности сигнала. 
    # Основная задача: охарактеризовать фильтр и отделить исходную часть
    # (на примере с голосом человека – охарактеризовать вокальный тракт).
    mfcc = librosa.feature.mfcc(y=sample, 
                                n_fft=n_fft, # размер фрейма
                                window='hann',  # оконная функция (windowing)
                                hop_length=int(n_fft*0.5), # размер перекрытия фреймов (overlapping)
                                sr=sample_rate, 
                                n_mfcc=20)
    features = np.mean(mfcc, axis=1)
    
    # librosa.feature.zero_crossing находит нулевые переходы для сигнала.
    zero_crossings = sum(librosa.zero_crossings(sample, pad=False))
    features = np.append(zero_crossings, features)
    
    # librosa.feature.spectral_centroid вычисляет спектральный центроид.
    # Каждый фрейм амплитудной спектрограммы нормализуется и обрабатывается как распределение по частотным элементам,
    # из которого извлекается среднее значение (центроид) для каждого фрейма.
    spec_cent = librosa.feature.spectral_centroid(y=sample,n_fft=n_fft, hop_length=int(n_fft*0.5), window='hann', sr=sample_rate).mean()
    features = np.append(spec_cent, features)
    
    # librosa.feature.spectral_flatness вычисляет cпектральную плоскостность.
    # Спектральная плоскостность - количественная мера того, насколько звук похож на шум, а не на тон.
    spec_flat = librosa.feature.spectral_flatness(y=sample,n_fft=n_fft, hop_length=int(n_fft*0.5), window='hann').mean()
    features = np.append(spec_flat, features)
    
    # librosa.feature.spectral_bandwith вычисляет спектральную полосу пропускания p-ого порядка.
    spec_bw = librosa.feature.spectral_bandwidth(y=sample,n_fft=n_fft, hop_length=int(n_fft*0.5), window='hann', sr=sample_rate).mean()
    features = np.append(spec_bw, features)
    
    # librosa.feature.spectral_rolloff вычисляет roll-off частоту для каждого фрейма.
    # Roll-off частота определяется как центральная частота для интервала спектрограммы.
    rolloff = librosa.feature.spectral_rolloff(y=sample, n_fft=n_fft, hop_length=int(n_fft*0.5), window='hann', sr=sample_rate).mean()
    features = np.append(rolloff, features)
    
    return features

signal_dataset_path = 'E:\\Training_Data\\'

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

features = pd.DataFrame()
results = []
features_sc = []
n_fft = 1024

def switch_signals_withoutcnt(coefs_file,temp_result,hf_signals_dict) :
    result_kls_label = 0
    if(temp_result == '0') :
        for (key, value) in hf_signals_dict.items() :    
            if(coefs_file.find(key) != -1) :
               result_kls_label = value[0];
    return result_kls_label
   

def get_signals_dict() :
    signals_dict = { 'human' : [1], 'spoof' : [2]  }
    return signals_dict


def get_codes_list(signals_dict) :
    codes_list = []
    for (key, value) in signals_dict.items() :  
        for i in range (0, len(value)) :
            codes_list.append(value[i])
    return codes_list



#fds = sorted(os.listdir(signal_dataset_path))

def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float32)

#def write_prediction(file_name):

def get_network():
    input_shape = (25,1)
    num_classes = 2
    keras.backend.clear_session()
    
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(128, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(64, activation="relu", input_shape=input_shape))
    model.add(keras.layers.Dense(num_classes, activation = "softmax"))
    model.compile(optimizer=keras.optimizers.Adam(1e-4), 
        loss="binary_crossentropy", 
        metrics=["accuracy"])
    
    return model

for signal_file in list_files(signal_dataset_path):
    if signal_file.endswith(('.wav')): 
        # librosa.load - загрузка аудио-файла.
        sample, sample_rate = librosa.load(signal_file)
        features = features.append(pd.Series(audio_features_extraction(sample, sample_rate, n_fft)), ignore_index=True)
        hf_signals_dict = get_signals_dict()
        for (key, value) in hf_signals_dict.items() :    
            if(signal_file.find(key) != -1) :
                results.append(value[0])


scaler = StandardScaler()
scaler.fit(features)
features_scaled = pd.DataFrame(scaler.transform(features))
#pickle.dump(scaler, open('scaler','wb'))
pickle.dump(scaler, open('F:/signalbase/speechbase/UrbanSound8K/save_nets/scaler_spoof_human.pkl','wb'))


features_scaled['result_kls_label'] = results

# конвертим данные под формат входных в Keras
X = np.array(features_scaled.loc[:, ~features_scaled.columns.isin(['result_kls_label'])])
y = np.array(features_scaled.result_kls_label.tolist())

# кодируем результаты, записанные буквами, в числа
#le = LabelEncoder()
#label_encoder = le.fit(y)
#label_encoded_y = le.transform(y) 

le = LabelEncoder()
label_encoded_y = to_categorical(le.fit_transform(y)) 

x_train, x_test, y_train, y_test = train_test_split(X, label_encoded_y, test_size=0.2, random_state = 42)




num_rows = 25
#num_columns = 1 убираем ведь у нас одно измерение
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_channels)

num_labels = label_encoded_y.shape[1]
filter_size = 2
#input_shape = (639,)
validation_data = x_test, y_test
num_epochs = 4000

#callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
model = get_network()
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=num_epochs,
                    validation_data=validation_data,
                    batch_size = 24, callbacks=[callback])

accuracies = []
#model.fit(x_train, y_train, epochs = 50, batch_size = 24, verbose = 0)
l, a = model.evaluate(x_test, y_test, verbose = 0)
accuracies.append(a)
print("Loss: {0} | Accuracy: {1}".format(l, a))


model.save('F:/signalbase/speechbase/UrbanSound8K/save_nets/keraskls_spoof_human.h5', include_optimizer=False)

testing_dataset_path = 'E:\\Testing_Data\\'
test_feature = pd.DataFrame()


for signal_file in list_files(testing_dataset_path):
    if signal_file.endswith(('.wav')): 
        # librosa.load - загрузка аудио-файла.
        sample, sample_rate = librosa.load(signal_file)
        test_feature = audio_features_extraction(sample, sample_rate, n_fft)
        test_feature_scaled = scaler.transform(test_feature.reshape(1,-1))
        test_feature_scaled = test_feature_scaled.reshape(1, num_rows, num_channels)
        predicted_vector = model.predict_classes(test_feature_scaled)
        predicted_class = le.inverse_transform(predicted_vector) 
        print("The predicted class is:", predicted_class[0], '\n') 

        predicted_proba_vector = model.predict_proba(test_feature_scaled) 
        predicted_proba = predicted_proba_vector[0]
        for i in range(len(predicted_proba)): 
            category = le.inverse_transform(np.array([i]))
            print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )            





