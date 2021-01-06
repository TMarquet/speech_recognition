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



############################
test_size = 1000
preprocessing = True
data_augmentation = False
noise = False
labels = ["yes", "no", "up", "down", "left",
"right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
"five", "six", "seven", "eight", "nine","unknown"]



def metrics(tp,fp,tn,fn,p,n):
    recall = tp/p
    fp_rate = fp/n
    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(p+n)
    return recall,fp_rate,precision,accuracy    



############################
        # MAIN
############################
test_data ,test_labels = get_test_data(test_size,data_augmentation,preprocessing,noise)
print('Test done on {} examples'.format(test_data.shape[0]))
directory_path = 'models'
for file in os.listdir(directory_path):
    print('Loading model :',file)
    model = load_model(directory_path+ '/'+file)


    true_positive = {}
    false_positive = {}
    true_negative = {}
    false_negative = {}
    confusion_matrix = {}
    P = {}
    N = {}
    for label in range(0,len(labels)):
        true_positive[label] = 0
        false_positive[label] = 0
        true_negative[label] = 0
        false_negative[label] = 0 
        confusion_matrix[label] = {}
        P[label] = 0
        N[label] = 0
        for l in range(0,len(labels)):
            confusion_matrix[label][l] = 0
    for i in range(len(test_data)):
        test = np.array([test_data[i]])
        prediction = model.predict(test)
        predicted_label = np.where(prediction == np.amax(prediction))
        
        true_label = np.where(test_labels[i] == np.amax(test_labels[i]))
        P[true_label[0][0]] += 1
        if predicted_label[1][0] == true_label[0][0]:
            true_positive[predicted_label[1][0]] += 1
            for label in range(0,len(labels)):
                if not label == predicted_label[1][0]:
                    true_negative[label] += 1            
        else:
            false_positive[predicted_label[1][0]] += 1
            false_negative[true_label[0][0]] += 1
            confusion_matrix[predicted_label[1][0]][true_label[0][0]] += 1
    for l , p in P.items():
        for label in range(0,len(labels)):
            if l != label:
                N[label] += p
    
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_p = 0
    total_n = 0
    for label in range(0,len(labels)):
        total_tp += true_positive[label]
        total_fp += false_positive[label]
        total_tn += true_negative[label]
        total_fn += false_negative[label]
        total_p += P[label]
        total_n += N[label]
        r,f,p,a = metrics(true_positive[label],false_positive[label],true_negative[label],false_negative[label],P[label],N[label])
        #print('For label {} : \n Recall is {} % \n False positive rate is : {} % \n Precision is : {} % \n Accuracy is : {} % '.format(label,r,f,p,a))
        
    r,f,p,a = metrics(total_tp,total_fp,total_tn,total_fn,total_p,total_n)
    print('In total : \n Recall is {} % \n False positive rate is : {} % \n Precision is : {} % \n Accuracy is : {} % '.format(r,f,p,a))