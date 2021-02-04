# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:51:33 2021

@author: kahg8
"""
import os
import sounddevice as sd
import numpy as np
import time
import scipy.io.wavfile as wav
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, LSTM, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from preprocessing import preprocess_live_data,count_down

#(gpus = tf.config.experimental.list_physical_devices('GPU')

# for gpu in gpus:

#     tf.config.experimental.set_memory_growth(gpu, True)

preprocessing = True
data_augmentation = False
fs=16000
duration = 1  # seconds
labels = ["yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine","silence","unknown"]

unknown_labels = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
"tree","wow"]


model1 = load_model('models/bests_silence0/mlp_mfcc_40epochs_50batchsize.h5')

i = ''
count  = 0
while True:
    
    i = input('Waiting for key press to record audio \n')
    if i == 'exit':
        break
    
    #count_down()
    
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='int16')
    
    print("Recording Audio")
    sd.wait()
    print("Audio recording complete")
    
    sd.play(myrecording, fs)
    sd.wait()
    
# =============================================================================
#     label = input()
#     if not label  in os.listdir('my_voice'):
#         os.mkdir('my_voice/'+label)
#     wav.write('my_voice/{}/{}.wav'.format(label,str(count)), fs, myrecording)
# =============================================================================
    inputs = preprocess_live_data(myrecording,fs)
   
    count +=1
    prediction1 = model1.predict(np.array([inputs]))
#    prediction2 = model2.predict(inputs)
    prediction = prediction1 
    predicted_label = np.where(prediction == np.amax(prediction))
    print("Final pred : ",labels[predicted_label[1][0]])

