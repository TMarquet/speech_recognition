# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:51:33 2021

@author: kahg8
"""

import sounddevice as sd
import numpy as np
import time
import scipy.io.wavfile as wav
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, LSTM, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from preprocessing import preprocess_live_data,count_down
preprocessing = True
data_augmentation = False
fs=16000
duration = 2  # seconds
labels = ["yes", "no", "up", "down", "left",
"right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
"five", "six", "seven", "eight", "nine","unknown"]

model1 = load_model('tested_models/ConvSpeechModel_20epochs_25batchsize_5layers_190000000training.h5')
model2 = load_model('models/cnn_best_50epochs_25batchsize.h5')
i = ''
while True:
    
    i = input('Waiting for key press to record audio + \n')
    if i == 'exit':
        break
    
    #count_down()
    
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float64')
    print("Recording Audio")
    sd.wait()
    print("Audio recording complete")
    
    sd.play(myrecording, fs)
    sd.wait()
    
    inputs = preprocess_live_data(myrecording,fs)
    
    prediction1 = model1.predict(inputs)
#    prediction2 = model2.predict(inputs)
    prediction = prediction1 
    predicted_label = np.where(prediction == np.amax(prediction))
    print("Final pred : ",labels[predicted_label[1][0]])

