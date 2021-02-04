# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:15:43 2021

@author: kahg8
"""

import argparse
import queue

import sys
from data_analysis import frame_generator
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import webrtcvad
import tensorflow as tf

from tensorflow.keras.models import load_model
from preprocessing import preprocess_live_data

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:

    tf.config.experimental.set_memory_growth(gpu, True)

labels = ["yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine","silence","unknown"]

unknown_labels = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
"tree","wow"]
model1 = load_model('models/bests_silence0/small_cnn_mfcc_60epochs_50batchsize.h5')

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')



args = parser.parse_args(remaining)





if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()
q_pred = queue.Queue(maxsize=31)
q_speak = queue.Queue()

vad = webrtcvad.Vad(int(3))

predicted_label = [[],[20]]

def get_best_chunk(data):
    best_chunk = []
    best_sum = 0
    for k in range(0,len(data)-16000,10):
        sub = data[k:k+16000]
        sum_sub = np.sum(abs(sub))
        if sum_sub > best_sum:
            best_sum = sum_sub
            best_chunk = sub
    return best_chunk


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:

    

   
    q.put(indata[::10, mapping])

    new_data = []
    for elem in indata:
        new_data.append(elem[0])
    frames_l = frame_generator(10, np.array(new_data), 16000)
    
    frames_l = list(frames_l) 

    speech = True
    
    for frame in frames_l:

        if not vad.is_speech(frame.bytes, 16000):
            speech = False
            break   


    
        
    was_speaking = False    
    if not q_speak.empty():
        was_speaking = q_speak.get_nowait()
    
    
    if speech and was_speaking :
        q_speak.put_nowait(True)
    elif speech and not was_speaking:
        q_speak.put_nowait(True)
    elif not speech and was_speaking:

        data = []
        
        for item in list(q_pred.queue):
            for elem in item:
                data.append( elem[0])


        data += new_data

        data = np.array(data)

        
        inputs = preprocess_live_data(data,16000)
            
        
        prediction = model1.predict(np.array([inputs]))
    #    prediction2 = model2.predict(inputs)
        
        predicted_label = np.where(prediction == np.amax(prediction))        
        print("Predicted : ",labels[predicted_label[1][0]])
  
        q_pred.queue.clear()        

    if q_pred.full():
        q_pred.get_nowait()
    q_pred.put(indata.copy())
 
    #print(indata.shape)
    


def update_plot(frame):
    global plotdata
    global ax

    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines 



try:

    length = int(args.window * 16000 / (1000 * 10))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(args.channels) > 1:
        ax.legend(['channel {}'.format(c) for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -1000, 1000))
    ax.set_yticks([0])

    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    label_to_plot = ax.text(3, 8,'silence', transform = ax.transAxes, style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})  
    stream = sd.InputStream(
        device=args.device,blocksize=500, channels=max(args.channels),
        samplerate=16000, callback=audio_callback,dtype=np.int16)

    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)
    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))