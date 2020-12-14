# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:19:20 2020

@author: kahg8
"""

import os
# Helper libraries
import numpy as np

from scipy.io import wavfile


from python_speech_features import mfcc

labels = ["yes", "no", "up", "down", "left",
"right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
"five", "six", "seven", "eight", "nine",'unknown']
unknown_labels = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
"tree","wow"]

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio


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
                



def get_windowed_signal(signal,len_w):
    windows = []
    n =int((1-len_w)*100+1)
    for i in range(0,n):
        windows.append(signal[10*i:30+10*i])
    windows *= np.hamming(len_w*1000)
    return windows
    
def get_filter_banks(signal,sample_rate,nfilt,NFFT):
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bins = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bins[m - 1])   # left
        f_m = int(bins[m])             # center
        f_m_plus = int(bins[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_banks = np.dot(signal, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return filter_banks

# def mfcc(audio,sample_rate,num_ceps=12):
#     NFFT = 512
#     cep_lifter = 22    
#     pre_emphasis = 0.97
#     signal = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])  
#     windowed = get_windowed_signal(signal, 0.03)
#     mag_w= np.absolute(np.fft.rfft(windowed, NFFT))  # Magnitude of the FFT
#     pow_w = ((1.0 / NFFT) * ((mag_w) ** 2))  # Power Spectrum
#     # audio = normalize_audio(audio)
#     filters = get_filter_banks(pow_w, sample_rate, 40, NFFT)
    
#     mfcc = fft.dct(filters, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]
    
#     (nframes, ncoeff) = mfcc.shape
#     n = np.arange(ncoeff)
#     lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
#     mfcc *= lift  #*
#     mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
#     # plt.figure(figsize=(15,5))

#     # plt.imshow(mfcc, aspect='auto', origin='lower');     
#     return mfcc


# def prepare_data(label,nb_files,offset):
#     used_files = 0
#     i = 0
#     data = []
#     all_files = os.listdir(label)

#     while used_files < nb_files :
#         path = os.path.join(label,all_files[offset + i])
#         # print('Loading file : ', all_files[used_files])
#         sample_rate, audio = wavfile.read(path)
#         mfcc_val = mfcc(audio,sample_rate)           
#         used_files +=1
        
        
#     return data


def get_training_data(training_size,validation_size,data_augmentation,preprocessing):
    test_files = np.loadtxt('testing_list.txt', dtype=str)
    validation_files = np.loadtxt('validation_list.txt', dtype=str)
    all_labels = labels[:len(labels)-1] + unknown_labels

    t_per_label = training_size // len(labels)
    v_per_label = validation_size // len(labels)
    t_remains = training_size % len(labels)
    v_remains = validation_size % len(labels)
    t_unknown_n = t_per_label + t_remains
    v_unknown_n = v_per_label + v_remains    
    
    t_per_u_label = t_unknown_n // len(unknown_labels)
    v_per_u_label = v_unknown_n // len(unknown_labels)
    
    training_data = []
    validation_data = []
    
    
    for label in sorted(all_labels) :        
        all_files = os.listdir(label)
        print('Processing label for training: ',label)
        count_training_file = 0
        
        last_audio1 = np.array([])
        for file in all_files :
            full_file_name = label + '/' + file
            
            if not (full_file_name in test_files) and not (full_file_name in validation_files) :
                path = os.path.join(label,file)
                sample_rate, audio = wavfile.read(path)
                if len(audio) == 16000 :
                    audio_to_process = [audio]
                    if data_augmentation:
                        if last_audio1.size != 0:
                            average = (last_audio1 + audio)/2
                            audio_to_process.append(average)
                        last_audio1 = audio
                    if preprocessing:
                        for signal in audio_to_process :
                            mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                            np.subtract(mfcc_feat,np.mean(mfcc_feat))
                            np.divide(mfcc_feat,mfcc_feat.std())
                            training_data.append((mfcc_feat,encode_label(label,labels,unknown_labels)))

                    else:
                        for signal in audio_to_process:
                            training_data.append((signal,encode_label(label,labels,unknown_labels))) 
                             
                       
                count_training_file += 1
            if (count_training_file == t_per_label and label in labels) or (count_training_file == t_per_u_label and label in unknown_labels):
                break
    count_validation_file = {}
    last_audio1 = {}
    for file in validation_files:
        label = file.split('/')[0]
        if not label in count_validation_file:
            count_validation_file[label] = 0
            print('Processing label for validation: ',label)
        if (count_validation_file[label] == v_per_label and label in labels) or (count_validation_file[label] == v_per_u_label and label in unknown_labels):
            continue  
        if not label in last_audio1:
            last_audio1[label] =  np.array([])
        path = file
        sample_rate, audio = wavfile.read(path)
        if len(audio) == 16000 :
            audio_to_process = [audio]
            if data_augmentation:
                if last_audio1[label].size != 0:
                    average = (last_audio1[label] + audio)/2
                    audio_to_process.append(average)
                last_audio1[label] = audio
            if preprocessing:
                for signal in audio_to_process :
                    mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                    np.subtract(mfcc_feat,np.mean(mfcc_feat))
                    np.divide(mfcc_feat,mfcc_feat.std())
                    validation_data.append((mfcc_feat,encode_label(label,labels,unknown_labels)))

            else:
                for signal in audio_to_process:
                    validation_data.append((signal,encode_label(label,labels,unknown_labels)))         
            count_validation_file[label] += 1
        
    
        
    np.random.shuffle(training_data)
    np.random.shuffle(validation_data)

    training_data_set =[]
    training_label_set = []
    for elem in training_data :
        training_data_set.append(elem[0].tolist())
        training_label_set.append(elem[1])
        
    validation_data_set =[]
    validation_label_set = []
    for elem in validation_data :
        validation_data_set.append(elem[0].tolist())
        validation_label_set.append(elem[1])
    print('Loading and preprocessing of the training data_set : Done !')
    
    return np.array(training_data_set),np.array(training_label_set),np.array(validation_data_set),np.array(validation_label_set)


def get_test_data(test_size,data_augmentation,preprocessing):
    test_files = np.loadtxt('testing_list.txt', dtype=str)


    per_label = test_size // len(labels)
    remains = test_size % len(labels)
    unknown_n = per_label + remains
    per_u_label = unknown_n // len(unknown_labels)
    
    test_data = []
    count_test_file = {}
    last_audio1 = {}
    for file in test_files:
        label = file.split('/')[0]
        if not label in count_test_file:
            count_test_file[label] = 0
            print('Processing label for testing: ',label)
        if (count_test_file[label] == per_label and label in labels) or (count_test_file[label] == per_u_label and label in unknown_labels):
            continue  
        if not label in last_audio1:
            last_audio1[label] =  np.array([])
        path = file
        sample_rate, audio = wavfile.read(path)
        if len(audio) == 16000 :
            audio_to_process = [audio]
            if data_augmentation:
                if last_audio1[label].size != 0:
                    average = (last_audio1[label] + audio)/2
                    audio_to_process.append(average)
                last_audio1[label] = audio
            if preprocessing:
                for signal in audio_to_process :
                    mfcc_feat = mfcc(signal,sample_rate,winlen=0.03)
                    np.subtract(mfcc_feat,np.mean(mfcc_feat))
                    np.divide(mfcc_feat,mfcc_feat.std())
                    test_data.append((mfcc_feat,encode_label(label,labels,unknown_labels)))

            else:
                for signal in audio_to_process:
                    test_data.append((signal,encode_label(label,labels,unknown_labels)))         
            count_test_file[label] += 1
    np.random.shuffle(test_data)

    test_data_set =[]
    test_label_set = []
    for elem in test_data :
        test_data_set.append(elem[0].tolist())
        test_label_set.append(elem[1])
    print('Loading and preprocessing of the testing data_set : Done !')
    
    return np.array(test_data_set),np.array(test_label_set)

# def prepare_data(label,n,offset,preprocessing,add_noise):
#     used_files = 0
#     i = 0
#     data = []
#     all_files = os.listdir(label)
#     # if add_noise :
#     #     noises = {}
#     #     sample_rate_n1,noises[1] = wavfile.read('_background_noise_/pink_noise.wav')
#     #     sample_rate_n2,noises[2] = wavfile.read('_background_noise_/white_noise.wav')
#     #     sample_rate_n3,noises[3] = wavfile.read('_background_noise_/doing_the_dishes.wav')
#     #     sample_rate_n4,noises[4] = wavfile.read('_background_noise_/exercise_bike.wav')
#     #     sample_rate_n5,noises[5] = wavfile.read('_background_noise_/running_tap.wav')
#     #     sample_rate_n6,noises[6] = wavfile.read('_background_noise_/dude_miaowing.wav')
#     last_audio1 = np.array([])
#     last_audio2 = np.array([])
#     last_audio3 = np.array([])
#     last_audio4 = np.array([])
#     last_audio5 = np.array([])
#     last_audio6 = np.array([])
#     last_audio7 = np.array([])
#     last_audio8 = np.array([])
#     while used_files < n :
#         path = os.path.join(label,all_files[offset + i])
#         # print('Loading file : ', all_files[used_files])
#         sample_rate, audio = wavfile.read(path)
              
#         # audio = normalize_audio(audio)
#         if len(audio) == 16000 :
#             if preprocessing:
#                 mfcc_feat = mfcc(audio,sample_rate,winlen=0.03)
#                 np.subtract(mfcc_feat,np.mean(mfcc_feat))
#                 np.divide(mfcc_feat,mfcc_feat.std())
#                 data.append((mfcc_feat,encode_label(label,labels)))
#                 if add_noise :
#                     if last_audio1.size != 0:
#                         average = (last_audio1 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio2.size != 0:
#                         average = (last_audio2 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio3.size != 0:
#                         average = (last_audio1 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio4.size != 0:
#                         average = (last_audio2 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio5.size != 0:
#                         average = (last_audio5 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio6.size != 0:
#                         average = (last_audio6 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio7.size != 0:
#                         average = (last_audio7 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio8.size != 0:
#                         average = (last_audio8 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     last_audio8 = last_audio7
#                     last_audio7 = last_audio6
#                     last_audio6 = last_audio5
#                     last_audio5 = last_audio4
#                     last_audio4 = last_audio3
#                     last_audio3 = last_audio2
#                     last_audio2 = last_audio1
                                        
                        
                            

#             else:
#                 data.append((audio,encode_label(label,labels)))
#                 if add_noise:
                    
#                     if last_audio1.size != 0:
#                         average = (last_audio1 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio2.size != 0:
#                         average = (last_audio2 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio3.size != 0:
#                         average = (last_audio1 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio4.size != 0:
#                         average = (last_audio2 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio5.size != 0:
#                         average = (last_audio5 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio6.size != 0:
#                         average = (last_audio6 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio7.size != 0:
#                         average = (last_audio7 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     if last_audio8.size != 0:
#                         average = (last_audio8 + audio)/2                    
#                         mfcc_feat_average = mfcc(average,sample_rate,winlen=0.03)
#                         np.subtract(mfcc_feat_average,np.mean(mfcc_feat_average))
#                         np.divide(mfcc_feat_average,mfcc_feat_average.std())
#                         data.append((mfcc_feat_average,encode_label(label,labels)))
#                     last_audio8 = last_audio7
#                     last_audio7 = last_audio6
#                     last_audio6 = last_audio5
#                     last_audio5 = last_audio4
#                     last_audio4 = last_audio3
#                     last_audio3 = last_audio2
#                     last_audio2 = last_audio1
#                     last_audio1 = audio  
                
#             used_files +=1
#         i += 1
#     return i , data

# def prepare_data_test(label,n,offset):
#     used_files = 0
#     i = 0
#     data = []
#     all_files = os.listdir(label)
#     while used_files < n :
#         path = os.path.join(label,all_files[-(offset + i)])
#         # print('Loading file : ', all_files[used_files])
#         sample_rate, audio = wavfile.read(path)        
#         # audio = normalize_audio(audio)        
        
#         if len(audio) == 16000 :
#             mfcc_feat = mfcc(audio,sample_rate,winlen=0.03)
#             np.subtract(mfcc_feat,np.mean(mfcc_feat))
#             np.divide(mfcc_feat,mfcc_feat.std())
#             data.append((mfcc_feat,encode_label(label,labels)))
#             used_files +=1
#         i += 1
#     return data









# def get_training_data(training_size,validation_size,preprocessing,add_noise):
#     t_per_label = training_size // len(labels)
#     v_per_label = validation_size // len(labels)
#     t_remains = training_size % len(labels)
#     v_remains = validation_size % len(labels)
#     t_unknown_n = t_per_label + t_remains
#     v_unknown_n = v_per_label + v_remains
#     training_data = []
#     validation_data = []
#     for label in labels :
#         print('Prepare : ',label)
#         if label != 'unknown' :
#             offset, t_new_data = prepare_data(label,t_per_label,0,preprocessing,add_noise)
#             training_data += t_new_data
#             offset, v_new_data = prepare_data(label,v_per_label,offset,preprocessing,False)
#             validation_data += v_new_data
#         else:
#             dict_offset ={}
#             for j in range(t_unknown_n):
#                 l = np.random.choice(unknown_labels)
#                 if l in dict_offset :
#                     offset = dict_offset[l]
#                 else:
#                     offset = 0
#                 offset_t, t_new_data = prepare_data(l,1,offset,preprocessing,add_noise)
#                 training_data += t_new_data
#                 offset += offset_t
#                 dict_offset[l] = offset
            
#             for j in range(v_unknown_n):
#                 l = np.random.choice(unknown_labels)
#                 if l in dict_offset :
#                     offset = dict_offset[l]
#                 else:
#                     offset = 0                
#                 offset_v, v_new_data = prepare_data(l,1,offset,preprocessing,add_noise)
#                 validation_data += v_new_data
#                 offset += offset_v
#                 dict_offset[l] = offset
                
   
#     np.random.shuffle(training_data)
#     np.random.shuffle(validation_data)

#     training_data_set =[]
#     training_label_set = []
#     for elem in training_data :
#         training_data_set.append(elem[0].tolist())
#         training_label_set.append(elem[1])
        
#     validation_data_set =[]
#     validation_label_set = []
#     for elem in validation_data :
#         validation_data_set.append(elem[0].tolist())
#         validation_label_set.append(elem[1])
#     print('Loading and preprocessing of the training data_set : Done !')
    
#     return np.array(training_data_set),np.array(training_label_set),np.array(validation_data_set),np.array(validation_label_set)


# def get_test_data(test_size):
#     per_label = test_size // len(labels)
#     remains = test_size % len(labels)
#     unknown_n = per_label + remains
#     test_data = []
#     for label in labels :
#         print('Prepare : ',label)
#         if label != 'unknown' :
#             test_data += prepare_data_test(label,per_label,0)
#         else:
#             for i in range(unknown_n):
#                 l = np.random.choice(unknown_labels)
#                 test_data += prepare_data_test(l, 1,i)
#     np.random.shuffle(test_data)

#     test_data_set =[]
#     test_label_set = []
#     for elem in test_data :
#         test_data_set.append(elem[0].tolist())
#         test_label_set.append(elem[1])
#     print('Loading and preprocessing of the testing data_set : Done !')
    
#     return np.array(test_data_set),np.array(test_label_set)










