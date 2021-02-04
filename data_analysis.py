# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:02:28 2021

@author: kahg8
"""
import webrtcvad
import os
# Helper libraries
import numpy as np
import time
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
from python_speech_features import mfcc,ssc
from matplotlib import cm
labels = ["yes", "no", "up", "down", "left",
"right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
"five", "six", "seven", "eight", "nine",'silence','unknown']
unknown_labels = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
"tree","wow"]
all_labels = labels[:len(labels)-2]  + unknown_labels


import collections
import contextlib
import sys
import wave

import webrtcvad


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def autolabel(rects,ax,test = False):
    """Attach a text label above each bar in *rects*, displaying its height."""
    up = 3
    if test:
        up = 15
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, up),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def get_minimal_window(signal,sample_rate):
    vad = webrtcvad.Vad(1)
    frame_duration = 10  # ms
    frames = frame_generator(frame_duration, signal, sample_rate)   
    frames = list(frames)

    for frame in frames:
        vad.is_speech(frame.bytes, sample_rate)

        

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def observe_training_data():
    all_labels = labels[:len(labels)-2] 
    test_files = np.loadtxt('testing_list.txt', dtype=str)
    validation_files = np.loadtxt('validation_list.txt', dtype=str) 
     
    total = 0
    count = []
    count2 = []
    min_audio = 99999999
    for label in sorted(all_labels):
        all_files = os.listdir(label)
        count_temp = 0
        count_temp2 = 0
        for file in all_files:
            path = label+'/'+file
            sample_rate, audio = wavfile.read(path)
            min_audio = min(min_audio,len(audio))
            count_temp2 += 1
            if not file in validation_files and not file in test_files and len(audio) == 16000:

                count_temp += 1
                total += 1
        count.append(count_temp)
        count2.append(count_temp2)
    for label in sorted(unknown_labels):
        all_files = os.listdir(label)
        count_temp = 0
        count_temp2 = 0
        for file in all_files:
            path = label+'/'+file
            sample_rate, audio = wavfile.read(path)
            
            min_audio = min(min_audio,len(audio))
            count_temp2 += 1
            if not file in validation_files and not file in test_files and len(audio) == 16000:
                
                count_temp += 1
                total += 1
        count.append(count_temp)
        count2.append(count_temp2)
    print(count)  
    print(total)
    print(min_audio)
    width = 5

    return count, count2


          
    
def observe_validation_data():
    all_labels = labels[:len(labels)-2] 

    validation_files = np.loadtxt('validation_list.txt', dtype=str) 
    
     
    count = {}
    total = 0 
    count_2 = {}
    

    for file in validation_files:
        label = file.split("/")[0]
        if not label in count:
            count[label] = 0
            count_2[label] = 0
        sample_rate, audio = wavfile.read(file)
        count_2[label] +=1
        if len(audio) == 16000:

            count[label] += 1
            total += 1
    
    
    print(count)  
    print(total)
    width = 5
    count_l = []
    count_l2 = []
    
    
    
    r_label = all_labels + unknown_labels
    for l in r_label:
        count_l.append(count[l])
        count_l2.append(count_2[l])
    return count_l, count_l2
    
def observe_test_data():
    all_labels = labels[:len(labels)-2] 

    validation_files = np.loadtxt('testing_list.txt', dtype=str) 
    
     
    count = {}
    total = 0 
    count_2 = {}
    

    for file in validation_files:
        label = file.split("/")[0]
        if not label in count:
            count[label] = 0
            count_2[label] = 0
        sample_rate, audio = wavfile.read(file)
        count_2[label] +=1
        if len(audio) == 16000:

            count[label] += 1
            total += 1
    
    
    print(count)  
    print(total)
    
    count_l = []
    count_l2 = []
    r_label = all_labels + unknown_labels
    for l in r_label:
        count_l.append(count[l])
        count_l2.append(count_2[l])
    return count_l,count_l2

def print_hist():
    count_t1, count_t2 = observe_training_data()
    count_v1, count_v2 = observe_validation_data()
    count_tt1, count_tt2 = observe_test_data()
    
    all_labels = labels[:len(labels)-2] 
    
    fig, ax = plt.subplots(2,1,sharey=True)
    x = np.arange(0,len(all_labels+ unknown_labels)*40,step = 40)
    # the histogram of the data
    width = 10
    
    ax[0].set_xlabel('Labels')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(all_labels + unknown_labels)    
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].set_ylabel('Number of examples')
    ax[0].set_title('Histogram of the number of file of 1s per label')
    
    
    ax[1].set_xlabel('Labels')
    
    
    rectst1 = ax[0].bar(x-width, count_t1, width,label = 'training set',align = 'center')
    rectst2 = ax[1].bar(x-width, count_t2, width,label = 'training set',align = 'center')
    rectsv1 = ax[0].bar(x, count_v1, width,label = 'validation set',align = 'center')
    rectsv2 = ax[1].bar(x, count_v2, width,label = 'validation set',align = 'center')
    rectstt1 = ax[0].bar(x+width, count_tt1, width,label = 'test set',align = 'center')
    rectstt2 = ax[1].bar(x+width, count_tt2, width,label = 'test set',align = 'center')
    
    
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(all_labels + unknown_labels)    
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1].set_ylabel('Number of examples')
    ax[1].set_ylim(top = 2750)
    ax[1].set_title('Histogram of the number of file available per label')
    ax[0].legend()
    ax[1].legend()
    
    
    
    
    
    autolabel(rectst1,ax[0])
    autolabel(rectst2,ax[1])
    autolabel(rectsv1,ax[0],test = True)
    autolabel(rectsv2,ax[1],test= True)
    autolabel(rectstt1,ax[0])
    autolabel(rectstt2,ax[1])

# observe_training_data()

# i= 4
# j = 1
# fig, ax = plt.subplots(i,j)
# count = 0
# count_r = 0
# label = 'stop'
# for k in range(45,45+i*j):
#     print((count//j,count%j))

#     path= os.listdir(label)[k]
#     file =label+'/'+path
#     sample_rate, audio = wavfile.read(file)
#     mfcc_data = ssc(audio)
#     mfcc_data = np.subtract(mfcc_data,np.mean(mfcc_data))
#     mfcc_data = np.divide(mfcc_data,mfcc_data.std())    
#     mfcc_data= np.swapaxes(mfcc_data, 0 ,1)
#     get_minimal_window(audio, sample_rate)

#     cax = ax[count%i].imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
#     # audio = np.subtract(audio,np.mean(audio))
#     # audio = np.divide(audio,audio.std())    
#     # signal = list(audio)
#     # if len(audio) < 16000:
#     #     signal +=  [0]*(16000 - len(audio))
#     # ax[count // j][count%j].plot(range(16000),signal)
    

#     count+=1
# plt.show()

# def main():
    
#     for label in all_labels:
#         print('Writing label :',label)
#         os.mkdir(label+'_cut/')
#         for file in os.listdir(label):
            
#             audio, sample_rate = read_wave(label+'/'+file)
#             vad = webrtcvad.Vad(int(3))
#             frames = frame_generator(30, audio, sample_rate)
#             frames = list(frames)
#             segments = vad_collector(sample_rate, 30, 300, vad, frames)

                
            
#             for i, segment in enumerate(segments):
#                 path = label+'_cut/'+file 
                
#                 write_wave(path, segment, sample_rate)
#                 if i == 1:
#                     print('Attention double morceau',path)
                
                
            
        

# main()