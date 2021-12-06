# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:50:45 2020

@author: kahg8
"""
import tensorflow as tf

# Helper libraries
import numpy as np



from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import RMSprop,Adagrad,Adam
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import TimeDistributed,Concatenate,Flatten, Dense, Input,Reshape,Permute, Lambda,Conv2D, Conv1D, MaxPooling1D,MaxPooling2D, GlobalMaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, LSTM, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from preprocessing import *
from tensorflow.keras.regularizers import l2
import pickle


tf.random.set_seed(7)
np.random.seed(7)
##############################################

training_size = 'all'

nb_epochs = 20
batch = 800
nb_layers = 5
num_ceps = 13
data_type = 'mfcc'
network_type = 'small_cnn'

data_augmentation = False

add_silence = False

add_noise = False

use_cut = False

if data_type == 'mfcc':
    
    use_raw = False
    
    use_mfcc = True
    
    use_ssc = False
elif data_type == 'ssc':
    use_raw = False
    
    use_mfcc = False
    
    use_ssc = True
else:
    use_raw = True
    
    use_mfcc = False
    
    use_ssc = False


if network_type == 'cnn':
    
    use_cnn = True
    
    use_small_cnn = False
    
    use_lstm = False
    
    use_lstm_cnn = False
elif network_type == 'small_cnn':
    use_cnn = False

    use_small_cnn = True
    
    use_lstm = False
    use_lstm_cnn = False
elif network_type == 'lstm':
    use_cnn = False
    
    use_small_cnn = False
    
    use_lstm = True
    use_lstm_cnn = False
elif network_type == 'lstm_cnn':
    use_lstm_cnn = True
    use_cnn = False
    
    use_small_cnn = False
    
    use_lstm = False


else:
    use_cnn = False
    
    use_small_cnn = False
    
    use_lstm = False
    use_lstm_cnn = False






if add_silence:

    labels = ["yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine","silence","unknown"]
else:
    labels = ["yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine","unknown"]   
unknown_labels = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
"tree","wow"]
# Adding backgroud noise after

##############################################
            # MODEL CREATION
##############################################


def create_model_cnn(data_type,layer_nb = 5, learning_rate = 0.0001,n_label = len(labels),dense_units=4096):        
    model = tf.keras.Sequential(name='cnn_'+data_type)
    
    if data_type == 'mfcc':
        input_shape = (98,13)
    if data_type == 'ssc' :
        input_shape = (98,26)
    
    # Convolution Blocks
    # Block 1
    model.add(Conv1D(64, 3, padding='same', name='block1_conv',input_shape = input_shape))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block1_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block1_pool'))  
    
    # Block 2
    model.add(Conv1D(128, 3, padding='same', name='block2_conv'))   
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block2_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block2_pool'))   
    
    # Block 3
    model.add(Conv1D(256, 3, padding='same', name='block3_conv'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block3_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block3_pool'))
    
    # Block 4
    model.add(Conv1D(512, 3, padding='same', name='block4_conv'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block4_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block4_pool'))
    
    
    # Block 5
    model.add(Conv1D(512, 3, padding='same', name='block5_conv'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block5_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling1D(2, strides=2, name='block5_pool'))
   
    
   # End convolution
   
    model.add(Dropout(0.5,name = 'Dropout'))
    
    model.add(Flatten(name='flatten'))    
    # Two Dense layers
    
    model.add(Dense(dense_units, name='fc1'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block_batchnorm_dense1'))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(Dropout(0.5))
        
    model.add(Dense(dense_units, name='fc2'))
    model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
    model.add(BatchNormalization(name='block_batchnorm_dense2'))
    model.add(tf.keras.layers.Activation('relu'))       
    
    model.add(Dense(n_label, activation='softmax', name='predictions'))
    
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model




def combined_model(layer_nb = 5,input_shape = (2,len(labels)), learning_rate = 0.00001,n_label = len(labels),dense_units=4084):   
    model = tf.keras.Sequential(name='combined')    
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

def create_model_mlp(data_type,layer_nb = 5,input_shape = (98,13), learning_rate = 0.00001,n_label = len(labels),dense_units=256):   
    model = tf.keras.Sequential(name='mlp_'+ data_type) 
    
    if data_type == 'mfcc':
        input_shape = (98,13)
    if data_type == 'ssc' :
        input_shape = (98,26)
        
    model.add(Dense(dense_units, name='fc1',input_shape = input_shape))
    model.add(BatchNormalization(name='block1_batchnorm'))
    model.add(tf.keras.layers.Activation('relu'))
    
    for i in range(2,layer_nb):
        model.add(Dense(dense_units, name='fc'+str(i)))
        model.add(BatchNormalization(name='block{}_batchnorm'.format(str(i))))
        model.add(tf.keras.layers.Activation('relu'))  
    
    model.add(Flatten(name='flatten'))

    model.add(Dense(n_label, activation='softmax', name='predictions'))
    optimizer = RMSprop(lr=learning_rate)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model






def create_model_lstm(data_type,cnn =False,layer_nb  = 5,classes = len(labels),lstm_nodes=64 , learning_rate = 0.0001):    
    
    
    if data_type == 'mfcc':
        input_shape = (98,13) if not cnn else (98,13,1)
    if data_type == 'ssc' :
        input_shape = (98,26) if not cnn else (98,26,1)

    model = Sequential(name = 'lstm_{}'.format('' if not cnn else 'cnn_')+ data_type)
    model.add(Input(shape = input_shape))
    
    if cnn:
        cnn = Sequential(name= 'cnn_entry_'+data_type)
            
        cnn.add(Conv1D(22, 3, padding='same', name='conv1'))
        cnn.add(BatchNormalization(name = 'batch_norm1'))
        cnn.add(tf.keras.layers.Activation('relu'))
        
        cnn.add(Conv1D(44, 3, padding='same', name='conv2'))
        cnn.add(BatchNormalization(name = 'batch_norm2'))
        cnn.add(tf.keras.layers.Activation('relu'))
        
        cnn.add(Conv1D(22, 3, padding='same', name='conv3'))
        cnn.add(BatchNormalization(name = 'batch_norm3'))
        cnn.add(tf.keras.layers.Activation('relu'))
        cnn.add(AveragePooling1D(2, strides=2, name='pooling'))
               
        model.add(TimeDistributed(cnn))
        model.add(Reshape((98,-1)))
            
    
    if layer_nb == 1:
        model.add(LSTM(lstm_nodes,name = 'lstm_entry'))
        model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
        model.add(BatchNormalization(name='block1_batchnorm'))

    else:
        model.add(LSTM(lstm_nodes, return_sequences=True,name = 'lstm_entry'))
        model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
        model.add(BatchNormalization(name='block1_batchnorm'))
        for i in range(2, layer_nb):
            
            model.add(LSTM(lstm_nodes, return_sequences=True,name = 'lstm_{}'.format(i)))
            model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
            model.add(BatchNormalization(name='block{}_batchnorm'.format(str(i))))
        model.add(LSTM(lstm_nodes,name = 'lstm_out'))
        model.add(Lambda(lambda x: K.l2_normalize(x,axis=1)))
        model.add(BatchNormalization(name='blockout_batchnorm'))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax', name='predictions'))
    
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def create_model_small_cnn(data_type,labels = len(labels) , learning_rate = 0.0001,dense_units = 200):    
    if data_type == 'mfcc':
        input_shape = (98,13)
    if data_type == 'ssc' :
        input_shape = (98,26)
        
    in1 = Input(shape=input_shape)
    conv = Conv1D(kernel_size=34, strides=17, filters=4, activation='relu', padding='same')(in1)
    bn = BatchNormalization()(conv)

    flatten = Flatten()(bn)


    x = Dense(50, activation='relu', kernel_initializer='random_uniform')(flatten)

    output = Dense(labels, activation='softmax')(x)
    model = Model(inputs = [in1],outputs = [output],name='cnn')
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model










##############################################
            # MAIN
##############################################

try:
    
    file = open(PATH_DATA+'{}_training_data_cut{}_raw{}_mfcc{}_ssc{}_silence{}_da{}_noise{}.npy'.format(training_size,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise),'rb')
    train_data = pickle.load(file)
    file.close()

    file = open(PATH_DATA+'{}_training_labels_cut{}_raw{}_mfcc{}_ssc{}_silence{}_da{}_noise{}.npy'.format(training_size,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise),'rb')
    train_label = pickle.load(file)
    file.close()
    
    file = open(PATH_DATA+'{}_validation_data_cut{}_raw{}_mfcc{}_ssc{}_silence{}_da{}_noise{}.npy'.format(training_size,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise),'rb')
    validation_data = pickle.load(file)
    file.close()    
    
    file = open(PATH_DATA+'{}_validation_labels_cut{}_raw{}_mfcc{}_ssc{}_silence{}_da{}_noise{}.npy'.format(training_size,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise),'rb')
    validation_label = pickle.load(file)
    file.close()
    
    print('Loaded using pickle')
    
except:
    train_data,train_label,validation_data,validation_label= get_training_data(training_size, labels, unknown_labels,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise)
    
    file = open(PATH_DATA+'{}_training_data_cut{}_raw{}_mfcc{}_ssc{}_silence{}_da{}_noise{}.npy'.format(training_size,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise),'wb')
    pickle.dump(train_data,file)
    file.close()

    file = open(PATH_DATA+'{}_training_labels_cut{}_raw{}_mfcc{}_ssc{}_silence{}_da{}_noise{}.npy'.format(training_size,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise),'wb')
    pickle.dump(train_label,file)
    file.close()
    
    file = open(PATH_DATA+'{}_validation_data_cut{}_raw{}_mfcc{}_ssc{}_silence{}_da{}_noise{}.npy'.format(training_size,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise),'wb')
    pickle.dump(validation_data,file)
    file.close()    
    
    file = open(PATH_DATA+'{}_validation_labels_cut{}_raw{}_mfcc{}_ssc{}_silence{}_da{}_noise{}.npy'.format(training_size,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise),'wb')
    pickle.dump(validation_label,file)
    file.close()
    

###############################################
        # Single train



if use_mfcc:
    print('Training on {} examples !'.format(train_data['mfcc'].shape))
    
    print('Validation on {} examples !'.format(validation_data['mfcc'].shape))
    if use_cnn:
        model = create_model_cnn('mfcc')
    elif use_small_cnn:
        model = create_model_small_cnn('mfcc')
    elif use_lstm:
        model = create_model_lstm('mfcc')
    elif use_lstm_cnn:
        model = create_model_lstm('mfcc',cnn = True)

    else:
        model = create_model_mlp('mfcc')
    

    model.fit(train_data['mfcc'] if not use_lstm_cnn else train_data['mfcc'].reshape((-1,98,13,1)),train_label['mfcc'],validation_data = (validation_data['mfcc'] if not use_lstm_cnn else validation_data['mfcc'].reshape((-1,98,13,1)) ,validation_label['mfcc']),epochs = nb_epochs,batch_size = batch)
    model.save(PATH_MODELS+'model')

if use_ssc:
    print('Training on {} examples !'.format(train_data['ssc'].shape[0]))
    
    print('Validation on {} examples !'.format(validation_data['ssc'].shape[0]))
    if use_cnn:
        model = create_model_cnn('ssc')
    elif use_small_cnn:
        model = create_model_small_cnn('ssc')
    elif use_lstm:
        model = create_model_lstm('ssc')
    elif use_lstm_cnn:
        model = create_model_lstm('ssc',cnn = True)

    else:
        model = create_model_mlp('ssc')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=PATH_MODELS + '{}_{}epochs_{}batchsize.h5'.format(model.name,nb_epochs,batch),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    model.fit(train_data['ssc'] if not use_lstm_cnn else train_data['ssc'].reshape((-1,98,26,1)),train_label['ssc'],validation_data = (validation_data['ssc'] if not use_lstm_cnn else validation_data['ssc'].reshape((-1,98,26,1)),validation_label['ssc']),epochs = nb_epochs,batch_size = batch,callbacks = model_checkpoint_callback)
 

if use_raw:
    print('Training on {} examples !'.format(train_data['raw'].shape))
    
    print('Validation on {} examples !'.format(validation_data['raw'].shape))
    
    
    if use_cnn:
        model = create_model_cnn('raw')
    elif use_small_cnn:
        model = create_model_small_cnn('raw')
    elif use_lstm:
        model = create_model_lstm('raw')
    elif use_lstm_cnn:
        model = create_model_lstm('raw',cnn = True)
    else:
        model = create_model_mlp('raw')
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=PATH_MODELS + '{}_{}epochs_{}batchsize.h5'.format(model.name,nb_epochs,batch),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    model.fit(train_data['raw'].reshape((-1,16000,1)),train_label['raw'],validation_data = (validation_data['raw'].reshape((-1,16000,1)),validation_label['raw']),epochs = nb_epochs,batch_size = batch,callbacks = model_checkpoint_callback)
  




















