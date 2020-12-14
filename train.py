# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:50:45 2020

@author: kahg8
"""
import tensorflow as tf
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input,Reshape,Permute, Lambda,Conv2D, Conv1D, MaxPooling1D,MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, LSTM, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from preprocessing import get_training_data
from kapre.composed import get_melspectrogram_layer
tf.random.set_seed(7)
##############################################

training_size = 190000000
validation_size = int(np.round(training_size/10))
preprocessing = True
data_augmentation = False
test_size = 100
nb_epochs = 10
batch_size = 25
nb_layers = 5
num_ceps = 13
labels = ["yes", "no", "up", "down", "left",
"right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
"five", "six", "seven", "eight", "nine","unknown"]

unknown_labels = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
"tree","wow"]
# Adding backgroud noise after

##############################################
            # MODEL CREATION
##############################################


def create_model_cnn(layer_nb = 5,input_shape = (98,num_ceps), learning_rate = 0.0001,mlp_nodes = 200,n_label = len(labels),dense_units=4096):    
    model = tf.keras.Sequential(name='cnn_best')
    
    # Convolution Blocks
    # Block 1
    model.add(Conv1D(64, 3, padding='same', name='block1_conv1',input_shape = input_shape))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block1_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block1_pool'))  
    model.add(Dropout(0.01))
    # Block 2
    model.add(Conv1D(128, 3, padding='same', name='block2_conv1'))    
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block2_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block2_pool'))   
    model.add(Dropout(0.03))
    # Block 3
    model.add(Conv1D(256, 3, padding='same', name='block3_conv1'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block3_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block3_pool'))
    model.add(Dropout(0.05))
    # Block 4
    model.add(Conv1D(512, 3, padding='same', name='block4_conv1'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block4_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block4_pool'))
    model.add(Dropout(0.1))
    
    # Block 5
    model.add(Conv1D(512, 3, padding='same', name='block5_conv1'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block5_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block5_pool'))
    model.add(Dropout(0.5))
    model.add(Flatten(name='flatten'))
    
    # Two Dense layers
    
    model.add(Dense(dense_units, name='fc1'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block6_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    
    
    
    model.add(Dense(dense_units, name='fc2'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block7_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))       

    # model.add(Dropout(0.5))
    
    model.add(Dense(n_label, activation='softmax', name='predictions'))

    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def convSpeech(layer_nb = 5,input_shape = (98,num_ceps),sampling_rate = 16000, learning_rate = 0.00001,n_label = len(labels)):    

    model = tf.keras.Sequential(name='ConvSpeechModel') 
    model.add(Input(input_shape))
    model.add(Reshape((98,num_ceps,1)))
    #model.add(Normalization2D(int_axis=0))
    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    # x = Reshape((94,80)) (x) #this is strange - but now we have (batch_size,
    # sequence, vec_dim)

    model.add(Conv2D(20, (5, 1), padding='same'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 1)))
    

    model.add(Conv2D(40, (3, 3), padding='same'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    

    model.add(Conv2D(80, (3, 3), padding='same'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    

    model.add(Conv2D(160, (3, 3), padding='same'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    

    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))

    model.add(Dense(n_label, activation='softmax'))

    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def create_model_mlp(layer_nb = 5,input_shape = (98,13), learning_rate = 0.00001,mlp_nodes = 200,n_label = 11,dense_units=128):   
    model = tf.keras.Sequential(name='cnn_best')    
    model.add(Dense(dense_units, name='fc1',input_shape = input_shape))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block1_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    
    for i in range(2,layer_nb):
        model.add(Dense(dense_units, name='fc'+str(i)))
        model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
        model.add(BatchNormalization(name='block{}_batchnorm'.format(str(i))))
        model.add(tf.keras.layers.Activation('relu'))  
    model.add(Flatten(name='flatten'))
    model.add(Dropout(0.5))    
    model.add(Dense(n_label, activation='softmax', name='predictions'))
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
##############################################
            # MAIN
##############################################
train_data,train_label,validation_data,validation_label = get_training_data(training_size,validation_size,data_augmentation,preprocessing)
print('Training on {} examples !'.format(train_data.shape[0]))

print('Validation on {} examples !'.format(validation_data.shape[0]))

if preprocessing :
    model = create_model_cnn()
else:
    model = convSpeech()
#model = create_model_mlp(5)
model.fit(train_data,train_label,validation_data=(validation_data,validation_label),epochs = nb_epochs,batch_size = batch_size)
model.save('models\{}epochs_{}batchsize_{}layers_{}training.h5'.format(nb_epochs,batch_size,nb_layers,training_size))