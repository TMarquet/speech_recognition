# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:19:20 2020

@author: kahg8
"""

import os
# Helper libraries
import numpy as np
import time
from scipy.io import wavfile

import random
from python_speech_features import mfcc,ssc
from tensorflow.keras.models import load_model

labels = ["yes", "no", "up", "down", "left",
"right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
"five", "six", "seven", "eight", "nine",'silence','unknown']
unknown_labels = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
"tree","wow"]

PATH_LIST = 'C:/Users/kahg8/Documents/GitHub/speech_recognition/lists/'
PATH_DATA = 'C:/Users/kahg8/Documents/GitHub/speech_recognition/data/'

# training_size = 'all'

# use_raw = False

# use_mfcc = True

# use_ssc = False

# use_cut = True

# add_silence = True

# data_augmentation = False

coef_silence = 1
coef_noise = 1

def normalize_audio(audio):
    n_audio = np.subtract(audio,np.mean(audio))
    n_audio = np.divide(n_audio,n_audio.std())  
    return n_audio

def pad_audio(signal):
    final_signal = signal
    
    n = len(signal)
    target_length = 16000
    to_add = target_length - n
    
    for i in range(to_add):
        if i <= to_add//2:
            final_signal.insert(0,0)
        else:
            final_signal.append(0)
    ret = np.array(final_signal,dtype = np.int16)
    
    return ret

def encode_label(x,y,z):
    encode = []
    for elem in y[:len(y)-1] :        
        if x == elem :
            encode.append(1)
        else:
            encode.append(0)
    if x in z:
        encode.append(1)
    else:
        encode.append(0)
    return encode


# max usable = 2062 files per label
def get_max_usable(labels):
    usable = 9999999
    l = labels[0:10]
    for label in l :
        all_files = os.listdir(label)
        count = 0
        print(label)
        for file in all_files:
            
            path = os.path.join(label,file)
        
            sample_rate, audio = wavfile.read(path)
            
            if len(audio)==16000:
                count +=1
        if count < usable :
            usable = count
    print(usable)
                
  

def preprocess_live_data_combined(signal,sample_rate):
    length_signal = len(signal)
    # if length_signal > 16000:
    #     best_chunk = []
    #     best_sum = 0
    
    #     for k in range(0,length_signal):
    #         sub = signal[k:k+16000]
    #         sum_sub = np.sum(abs(sub))
    #         if sum_sub > best_sum:
    #             best_sum = sum_sub
    #             best_chunk = sub
    #     # plt.plot(range(0,16000),best_chunk)
    #     # plt.show()        
    #     signal = best_chunk
    mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
    np.subtract(mfcc_feat,np.mean(mfcc_feat))
    np.divide(mfcc_feat,mfcc_feat.std())
    model_mfcc = load_model('models/cnn_mfcc_20epochs_50batchsize.h5')
    prediction_mfcc = model_mfcc.predict(np.array([mfcc_feat]))
    
    ssc_feat = ssc(signal,sample_rate,winlen=0.03)
    np.subtract(ssc_feat,np.mean(ssc_feat))
    np.divide(ssc_feat,ssc_feat.std())
    model_ssc = load_model('models/cnn_ssc_20epochs_50batchsize.h5')
    prediction_ssc = model_ssc.predict(np.array([ssc_feat])) 
    
    return np.concatenate((prediction_mfcc,prediction_ssc)) 
def preprocess_live_data(signal,sample_rate):
    length_signal = len(signal)
    # if length_signal > 16000:
    #     best_chunk = []
    #     best_sum = 0
    
    #     for k in range(0,length_signal):
    #         sub = signal[k:k+16000]
    #         sum_sub = np.sum(abs(sub))
    #         if sum_sub > best_sum:
    #             best_sum = sum_sub
    #             best_chunk = sub
    #     # plt.plot(range(0,16000),best_chunk)
    #     # plt.show()        
    #     signal = best_chunk
    mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
    np.subtract(mfcc_feat,np.mean(mfcc_feat))
    np.divide(mfcc_feat,mfcc_feat.std())

    
    # ssc_feat = ssc(signal,sample_rate,winlen=0.03)
    # np.subtract(ssc_feat,np.mean(ssc_feat))
    # np.divide(ssc_feat,ssc_feat.std())

    return mfcc_feat
def count_down():
    start = time.time()
    while(time.time() - start < 3):
        count = time.time()-start
        if count.is_integer():
            print(str(int(count))+' ! \n')

def make_training_list(labels,unknown_labels,training_size):
    all_labels = labels[:len(labels)-2] 
    test_files = np.loadtxt(PATH_LIST + 'testing_list.txt', dtype=str)
    validation_files = np.loadtxt(PATH_LIST + 'validation_list.txt', dtype=str) 
    training_files = []  
    total = 0
    if training_size == 'all':
        max_k = 2060
        max_un = 1000
        name = 'training_list_all.txt'
    else:
        max_k = 100
        max_un = 10
        name= 'training_list_small.txt'
        
    with open(PATH_DATA + name,'w') as f:
        for label in sorted(all_labels):
            all_files = os.listdir(label)
            count = 0
            for file in all_files:
                path = label+'/'+file
                sample_rate, audio = wavfile.read(path)
                if not file in validation_files and not file in test_files and len(audio) == 16000 and count < max_k:
                    f.write(path + '\n')
                    count += 1
                    total += 1
            print(count)

        for label in sorted(unknown_labels):
            all_files = os.listdir(label)
            count = 0
            for file in all_files:
                path = label+'/'+file
                sample_rate, audio = wavfile.read(path)
                if not file in validation_files and not file in test_files and len(audio) == 16000 and count < max_un:
                    f.write(path + '\n')
                    count += 1
                    total += 1
            print(count)
    print(total)            

def make_validation_list(labels,unknown_labels,training_size):
    all_labels = labels[:len(labels)-2] 

    validation_files = np.loadtxt(PATH_LIST + 'validation_list.txt', dtype=str) 
    training_files = []  
    count = {}
    total = 0 
    if training_size == 'all':
        max_k = 200
        max_un = 100
        name = 'validation_list_all.txt'
    else:
        max_k = 20
        max_un = 2  
        name = 'validation_list_small.txt'
    
    with open(PATH_LIST + name,'w') as f:
        for file in validation_files:
            label = file.split("/")[0]
            if not label in count:
                count[label] = 0
            sample_rate, audio = wavfile.read(file)
            if label in unknown_labels:
                max_label = max_un
            else:
                max_label = max_k
            
            if len(audio) == 16000 and count[label] < max_label:
                f.write(file + '\n')
                count[label] += 1
                total += 1
    print(count)  
    print(total)            
def get_training_data(training_size,labels,unknown_labels,use_cut, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise):
    
    if training_size == 'all':
        training_files = np.loadtxt(PATH_LIST + 'training_list_all.txt', dtype=str)
        validation_files = np.loadtxt(PATH_LIST + 'validation_list_all.txt', dtype=str)
    else:
        training_files = np.loadtxt(PATH_LIST + 'training_list_small.txt', dtype=str)
        validation_files = np.loadtxt(PATH_LIST + 'validation_list_small.txt', dtype=str)
    if not add_silence:
        labels = labels[0:20]+[labels[-1]]
    training_data = {'raw':[],'mfcc':[],'ssc':[]}
    validation_data = {'raw':[],'mfcc':[],'ssc':[]}
    noises = []
    for file in os.listdir(PATH_DATA + '_background_noise_'):
        if 'wav' in file:
            path = os.path.join(PATH_DATA + '_background_noise_',file)
            sample_rate, audio = wavfile.read(path)            
            noises.append(audio)    
    first = {}   
    rescale_l = []    
    for file in training_files:
        label = file.split('/')[0]
        name  = file.split('/')[1]
        if not label in first:
            first[label] = True
            print('Processing label for training: ',label)
            audio_for_average = np.array([])

        if use_cut and name in os.listdir(label+'_cut'):
                path = label + '_cut/'+ name
                sample_rate, audio = wavfile.read(path)
                     
        else:
            path = PATH_DATA + file
            sample_rate, audio = wavfile.read(path)
        if len(audio < 16000):
            audio = pad_audio(audio) 
        if add_noise:
            noise_type = random.randint(0, len(noises)-1)
            noise = noises[noise_type]
            window = random.randint(0, len(noise)-16000-1)

            noise_signal = coef_noise*noise[window:window+16000]  
            audio = audio + noise_signal
        
        audio_to_process = [audio]
        if data_augmentation:
            if audio_for_average.size != 0:
                average = (audio_for_average + audio)/2
                audio_to_process.append(average)
        data_to_add = {'raw':[],'mfcc':[],'ssc':[]}
        for signal in audio_to_process:                        
            if use_raw :
                normalized_signal = normalize_audio(signal)
                data_to_add['raw'].append(normalized_signal)
            if use_mfcc :
                mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                np.subtract(mfcc_feat,np.mean(mfcc_feat))
                np.divide(mfcc_feat,mfcc_feat.std())  
                data_to_add['mfcc'].append(mfcc_feat)
            if use_ssc:
                ssc_feat =  ssc(signal,sample_rate,winlen=0.03)
                np.subtract(ssc_feat,np.mean(ssc_feat))
                np.divide(ssc_feat,ssc_feat.std()) 
                data_to_add['ssc'].append(ssc_feat)
        for data_type , data in data_to_add.items():                    
            for elem_data in data:                        
                training_data[data_type].append((elem_data,encode_label(label,labels,unknown_labels)))                        
    first = {} 
    
    for file in validation_files:
        label = file.split("/")[0]
        if not label in first:
            first[label] = True
            print('Processing label for validation: ',label)
            audio_for_average = np.array([])

        if use_cut and name in os.listdir(label+'_cut'):
                path = label + '_cut/'+ name
                sample_rate, audio = wavfile.read(path)
                     
        else:
            path = PATH_DATA + file
            sample_rate, audio = wavfile.read(path)
        if len(audio < 16000):
            audio = pad_audio(audio) 
        if add_noise:
            noise_type = random.randint(0, len(noises)-1)
            noise = noises[noise_type]
            window = random.randint(0, len(noise)-16000-1)

            noise_signal = coef_noise*noise[window:window+16000]  
          
            audio = audio + noise_signal
        audio_to_process = [audio]

        data_to_add = {'raw':[],'mfcc':[],'ssc':[]}
        for signal in audio_to_process:                        
            if use_raw :
                normalized_signal = normalize_audio(signal)
                data_to_add['raw'].append(normalized_signal)
            if use_mfcc :
                mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                np.subtract(mfcc_feat,np.mean(mfcc_feat))
                np.divide(mfcc_feat,mfcc_feat.std())  
                data_to_add['mfcc'].append(mfcc_feat)
            if use_ssc:
                ssc_feat =  ssc(signal,sample_rate,winlen=0.03)
                np.subtract(ssc_feat,np.mean(ssc_feat))
                np.divide(ssc_feat,ssc_feat.std()) 
                data_to_add['ssc'].append(ssc_feat)
        for data_type , data in data_to_add.items():
            count_val = 0                    
            for elem_data in data:                        
                validation_data[data_type].append((elem_data,encode_label(label,labels,unknown_labels)))             
                count_val +=1
            
    if add_silence:      
        if training_size == 'all':
            nb_silence_to_add_t = 2060
            nb_silence_to_add_v = 206
        else:
            nb_silence_to_add_t = 200
            nb_silence_to_add_v = 20
    
    
    
    
        for i in range(nb_silence_to_add_t):
            
            silence_type = random.randint(0, len(noises)-1)
            noise = noises[silence_type]
            window = random.randint(0, len(noise)-16000-1)
            coef_silence = random.random()
            signal = coef_silence*noise[window:window+16000]
            data_to_add = {'raw':[],'mfcc':[],'ssc':[]}
            if use_raw :
                normalized_signal = normalize_audio(signal)
                data_to_add['raw'].append(normalized_signal)
            if use_mfcc :
                mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                np.subtract(mfcc_feat,np.mean(mfcc_feat))
                np.divide(mfcc_feat,mfcc_feat.std())  
                data_to_add['mfcc'].append(mfcc_feat)
            if use_ssc:
                ssc_feat =  ssc(signal,sample_rate,winlen=0.03)
                np.subtract(ssc_feat,np.mean(ssc_feat))
                np.divide(ssc_feat,ssc_feat.std()) 
                data_to_add['ssc'].append(ssc_feat)        
            for data_type , data in data_to_add.items():                    
                for elem_data in data:                        
                    training_data[data_type].append((elem_data,encode_label('silence',labels,unknown_labels)))        
    
    
        for i in range(nb_silence_to_add_v):
            
            silence_type = random.randint(0, len(noises)-1)
            noise = noises[silence_type]
            window = random.randint(0, len(noise)-16000-1)
            coef_silence = random.random()
            signal = coef_silence*noise[window:window+16000]
            data_to_add = {'raw':[],'mfcc':[],'ssc':[]}
            if use_raw :
                normalized_signal = normalize_audio(signal)
                data_to_add['raw'].append(normalized_signal)
            if use_mfcc :
                mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                np.subtract(mfcc_feat,np.mean(mfcc_feat))
                np.divide(mfcc_feat,mfcc_feat.std())  
                data_to_add['mfcc'].append(mfcc_feat)
            if use_ssc:
                ssc_feat =  ssc(signal,sample_rate,winlen=0.03)
                np.subtract(ssc_feat,np.mean(ssc_feat))
                np.divide(ssc_feat,ssc_feat.std()) 
                data_to_add['ssc'].append(ssc_feat)        
            for data_type , data in data_to_add.items():                    
                for elem_data in data:                        
                    validation_data[data_type].append((elem_data,encode_label('silence',labels,unknown_labels)))
                
    training_data_set = {'raw':[],'mfcc':[],'ssc':[]}
    training_data_label = {'raw':[],'mfcc':[],'ssc':[]}
    validation_data_set = {'raw':[],'mfcc':[],'ssc':[]}
    validation_data_label = {'raw':[],'mfcc':[],'ssc':[]}
    for data_type, data in training_data.items():
        np.random.shuffle(data)
        for elem in data:           
            training_data_set[data_type].append(elem[0].tolist())
            training_data_label[data_type].append(elem[1])
        training_data_set[data_type] = np.array(training_data_set[data_type])
        training_data_label[data_type] = np.array(training_data_label[data_type])
    for data_type, data in validation_data.items():
        np.random.shuffle(data)
        for elem in data:           
            validation_data_set[data_type].append(elem[0].tolist())
            validation_data_label[data_type].append(elem[1])            
        validation_data_set[data_type] = np.array(validation_data_set[data_type])
        validation_data_label[data_type] = np.array(validation_data_label[data_type])

    return training_data_set , training_data_label , validation_data_set, validation_data_label
    



def get_test_data(labels,unknown_labels,test_size, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise):
    
    
    
    test_files = np.loadtxt(PATH_LIST+ 'testing_list.txt', dtype=str)

    
    test_data = {'raw':[],'mfcc':[],'ssc':[]}
    
    noises = []
    for file in os.listdir(PATH_DATA+'_background_noise_'):
        if 'wav' in file:
            path = os.path.join(PATH_DATA+'_background_noise_',file)
            sample_rate, audio = wavfile.read(path)            
            noises.append(audio)    
    first = {}       
    for file in test_files:
        label = file.split('/')[0]
        if not label in first:
            first[label] = True
            print('Processing label for test: ',label)
            audio_for_average = np.array([])

        path = PATH_DATA +file
        sample_rate, audio = wavfile.read(path)
        if add_noise:
            noise_type = random.randint(0, len(noises)-1)
            noise = noises[noise_type]
            window = random.randint(0, len(noise)-16000-1)
            noise_signal = coef_noise*noise[window:window+16000]            
            audio = audio + noise_signal
        audio_to_process = [audio]
        if len(audio) < 16000:
            continue
        if data_augmentation:
            if audio_for_average.size != 0:
                average = (audio_for_average + audio)/2
                audio_to_process.append(average)
        data_to_add = {'raw':[],'mfcc':[],'ssc':[]}
        for signal in audio_to_process:                        
            if use_raw :
                normalized_signal = normalize_audio(signal)
                data_to_add['raw'].append(normalized_signal)
            if use_mfcc :
                mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                np.subtract(mfcc_feat,np.mean(mfcc_feat))
                np.divide(mfcc_feat,mfcc_feat.std())  
                data_to_add['mfcc'].append(mfcc_feat)
            if use_ssc:
                ssc_feat =  ssc(signal,sample_rate,winlen=0.03)
                np.subtract(ssc_feat,np.mean(ssc_feat))
                np.divide(ssc_feat,ssc_feat.std()) 
                data_to_add['ssc'].append(ssc_feat)
        for data_type , data in data_to_add.items():                    
            for elem_data in data:                        
                test_data[data_type].append((elem_data,encode_label(label,labels,unknown_labels)))                        

            
    if add_silence:        

        nb_silence_to_add_t = 200
   
        for i in range(nb_silence_to_add_t):
            silence_type = random.randint(0, len(noises)-1)
            noise = noises[silence_type]
            window = random.randint(0, len(noise)-16000-1)
            coef_silence = random.random()
            signal = coef_silence*noise[window:window+16000]
            data_to_add = {'raw':[],'mfcc':[],'ssc':[]}
            if use_raw :
                normalized_signal = normalize_audio(signal)
                data_to_add['raw'].append(normalized_signal)
            if use_mfcc :
                mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                np.subtract(mfcc_feat,np.mean(mfcc_feat))
                np.divide(mfcc_feat,mfcc_feat.std())  
                data_to_add['mfcc'].append(mfcc_feat)
            if use_ssc:
                ssc_feat =  ssc(signal,sample_rate,winlen=0.03)
                np.subtract(ssc_feat,np.mean(ssc_feat))
                np.divide(ssc_feat,ssc_feat.std()) 
                data_to_add['ssc'].append(ssc_feat)        
            for data_type , data in data_to_add.items():                    
                for elem_data in data:                        
                    test_data[data_type].append((elem_data,encode_label('silence',labels,unknown_labels)))        
    
    
                    
    test_data_set = {'raw':[],'mfcc':[],'ssc':[]}
    test_data_label = {'raw':[],'mfcc':[],'ssc':[]}

    for data_type, data in test_data.items():

        for elem in data:           
            test_data_set[data_type].append(elem[0].tolist())
            test_data_label[data_type].append(elem[1])
        test_data_set[data_type] = np.array(test_data_set[data_type])
        test_data_label[data_type] = np.array(test_data_label[data_type])


    return test_data_set , test_data_label











