# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:00:17 2020

@author: kahg8
"""

import tensorflow as tf
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, LSTM, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from preprocessing import get_test_data
from kapre.composed import get_melspectrogram_layer


############################
test_size = 100000000
preprocessing = True
data_augmentation = False


############################
        # MAIN
############################
test_data ,test_labels = get_test_data(test_size,data_augmentation,preprocessing)
print('Test done on {} examples'.format(test_data.shape[0]))

for file in os.listdir('models'):
    print('Loading model :',file)
    model = load_model('models/'+file)
    success = 0
    for i in range(len(test_data)):
        test = np.array([test_data[i]])
        prediction = model.predict(test)
        predicted_label = np.where(prediction == np.amax(prediction))
        
        true_label = np.where(test_labels[i] == np.amax(test_labels[i]))
        if predicted_label[1][0] == true_label[0][0]:
            success +=1
    
    print('Success rate of : {}%'.format(100*success/len(test_data)))