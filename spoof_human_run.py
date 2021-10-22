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
from keras.models import load_model


le = LabelEncoder()
labels = []
labels.append(1) 
labels.append(2) 
y = np.array(labels)

# кодируем результаты, записанные буквами, в числа
yy = to_categorical(le.fit_transform(y)) 

model = load_model('F:/signalbase/speechbase/UrbanSound8K/save_nets/keraskls_spoof_human.h5')


testing_dataset_path = 'E:\\Testing_Data\\test'
test_feature = pd.DataFrame()
scaler = pickle.load(open('F:/signalbase/speechbase/UrbanSound8K/save_nets/scaler_spoof_human.pkl','rb')) 

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
testing_dataset_path = 'E:\\Testing_Data\\'
n_fft = 1024
num_rows = 25
num_channels = 1
row = []


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

with open('F:/signalbase/speechbase/UrbanSound8K/save_nets/result_human_spoof.csv', 'w',newline='') as f:
    writer = csv.writer(f)
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
            
            row.append('<' + str(os.path.basename(signal_file)) + '>')
            row.append('<' + str(predicted_proba[0]) + '>')
            writer.writerow(row)
            row.clear()


