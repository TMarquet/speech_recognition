# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:00:17 2020

@author: kahg8
"""


import os
# Helper libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


from preprocessing import get_test_data



############################
test_size = 'all'
use_raw = False

use_mfcc = True

use_ssc = True

add_silence = True

add_noise = False

data_augmentation = False
labels = ["yes", "no", "up", "down", "left",
"right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four",
"five", "six", "seven", "eight", "nine","silence","unknown"]
unknown_labels = ["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila",
"tree","wow"]

def transform(l):
    n = len(labels[l])
    str_l =''
    max_n = 7
    to_add = max_n + 2 - n
    middle = to_add // 2
    for i in range(to_add):
        if not i == middle:
            str_l += ' '
        else:
            str_l += labels[l]
    return str_l

def cust_round(n):
    str_n = str(round(n,2))
    ret_s = str(round(n,2))
    if len(str_n) == 3:
        ret_s += "0"
    elif len(str_n) == 1:
        ret_s +=".00"
    return ret_s



def metrics(tp,fp,tn,fn):
    recall = tp/(tp+fn)
    if fp + fn != 0:
        fp_rate = fp/(fp+tn)
    else:
        fp_rate = 0
    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1 = 2/((1/recall)+(1/precision))
    return recall,fp_rate,precision,accuracy,f1    

def heatmap(data_mfcc, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    fig, ax = plt.subplots()

    # Plot the heatmap
    im1 = ax.imshow(data_mfcc, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im1, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel+ '_MFCC', rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data_mfcc.shape[1]))
    ax.set_yticks(np.arange(data_mfcc.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data_mfcc.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data_mfcc.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im1





def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts    


def hist(data_mfcc,data_ssc,list_label,name):
    
    

    
    x = np.arange(len(list_label))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x- width/2 , data_mfcc, width, label=name + ' MFCC')
    rects2 = ax.bar(x+ width/2 , data_ssc, width, label=name + ' SSC')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('labels')
    ax.set_title(name + ' by label')
    ax.set_xticks(x)
    ax.set_xticklabels(list_label)
    ax.legend()
    
    
    
    
    fig.tight_layout()
    
    plt.show()

############################
        # MAIN
############################


test_data ,test_labels  = get_test_data(labels, unknown_labels,test_size, use_raw, use_mfcc, use_ssc, add_silence, data_augmentation,add_noise)
print(test_data['mfcc'].shape)
print(test_data['ssc'].shape)
directory_path = 'models'
for file in os.listdir(directory_path):
    if not '.h5' in file:
        continue
    print('Loading model :',file)
    
    model_name = 'small_cnn_mfcc_60epochs_50batchsize.h5'
    model = load_model(directory_path+'/bests_silence0/' + model_name)
    model_name_ssc = 'small_cnn_ssc_80epochs_50batchsize.h5'
    model_ssc = load_model(directory_path+'/bests_silence0/' + model_name_ssc)

    true_positive = {}
    false_positive = {}
    true_negative = {}
    false_negative = {}
    confusion_matrix = {}
 
    true_positive_ssc = {}
    false_positive_ssc = {}
    true_negative_ssc = {}
    false_negative_ssc = {}
    confusion_matrix_ssc = {}    
 
    data_type = 'raw'
    if 'mfcc' in model_name:
        data_type = 'mfcc'
    if 'ssc' in model_name:
        data_type = 'ssc'
    
    
    
    somme_test = {}
    count = 0
    success = 0

    success_ssc = 0
    for label in range(0,len(labels)):
        true_positive[label] = 0
        false_positive[label] = 0
        true_negative[label] = 0
        false_negative[label] = 0 
        confusion_matrix[label] = {}
        true_positive_ssc[label] = 0
        false_positive_ssc[label] = 0
        true_negative_ssc[label] = 0
        false_negative_ssc[label] = 0 
        confusion_matrix_ssc[label] = {}
        for l in range(0,len(labels)):
            confusion_matrix[label][l] = 0
            confusion_matrix_ssc[label][l] = 0
    for i in range(len(test_data['mfcc'])):
        test_mfcc = test_data['mfcc'][i]
        test_ssc = test_data['ssc'][i]
        count +=1
        # test2 = test_data2[i]
        # test3 = test_data3[i]
        #inputs = preprocess_live_data(test,16000)
        # prediction_mfcc = model.predict(np.array([test_mfcc]).reshape(-1,98,13,1))
        # prediction_ssc = model_ssc.predict(np.array([test_ssc]).reshape(-1,98,26,1))
        prediction_mfcc = model.predict(np.array([test_mfcc]))
        prediction_ssc = model_ssc.predict(np.array([test_ssc]))
        
        predicted_label_mfcc = np.where(prediction_mfcc == np.amax(prediction_mfcc))
        predicted_label_ssc = np.where(prediction_ssc == np.amax(prediction_ssc))
        true_label = np.where(test_labels['mfcc'][i] == np.amax(test_labels['mfcc'][i]))

        if predicted_label_mfcc[1][0] == true_label[0][0]:
            
            true_positive[predicted_label_mfcc[1][0]] += 1
            success +=1
            for label in range(0,len(labels)):
                if not label == predicted_label_mfcc[1][0]:
                    true_negative[label] += 1            
        else:
           
            false_positive[predicted_label_mfcc[1][0]] += 1
            false_negative[true_label[0][0]] += 1
            confusion_matrix[predicted_label_mfcc[1][0]][true_label[0][0]] += 1

        if predicted_label_ssc[1][0] == true_label[0][0]:
            
            true_positive_ssc[predicted_label_ssc[1][0]] += 1
            success_ssc +=1
            for label in range(0,len(labels)):
                if not label == predicted_label_ssc[1][0]:
                    true_negative_ssc[label] += 1            
        else:
           
            false_positive_ssc[predicted_label_ssc[1][0]] += 1
            false_negative_ssc[true_label[0][0]] += 1
            confusion_matrix_ssc[predicted_label_ssc[1][0]][true_label[0][0]] += 1

    

        #print('For label {} : \n Recall is {} % \n False positive rate is : {} % \n Precision is : {} % \n Accuracy is : {} % '.format(label,r,f,p,a))
       
    

    
    print('Success rate MFCC : {} %'.format( 100*success/count))
    print('Success rate SSC : {} %'.format( 100*success_ssc/count))
    matrix = []
    matrix_ssc = []
    unchanged = []
    unchanged_ssc = []
    for k1 , v1 in confusion_matrix.items():
        line = []
        line_unchanged = []
        somme = 0
        for k2 , v2 in v1.items():
            if false_positive[k1] != 0:
                line.append(100*v2/(false_positive[k1]))
                line_unchanged.append(int(v2))
            else:
                line.append(0)
                line_unchanged.append(int(0))
        matrix.append(line)
        unchanged.append(line_unchanged)
    for k1 , v1 in confusion_matrix_ssc.items():
        line = []
        line_unchanged = []
        somme = 0
        for k2 , v2 in v1.items():
            if false_positive_ssc[k1] != 0:
                line.append(100*v2/(false_positive_ssc[k1]))
                line_unchanged.append(int(v2))
            else:
                line.append(0)
                line_unchanged.append(int(0))
        matrix_ssc.append(line)
        unchanged_ssc.append(line_unchanged)


    R = []
    FPR = []
    P = []
    A = []
    F1 = []

    R_ssc = []
    FPR_ssc = []
    P_ssc = []
    A_ssc = []
    F1_ssc = []
    print('|================================================================|')    
    print('|        | FPR  |  R   |  A   |  P   | FPR  |  R   |  A   |  P   |')
    print('|========|-----------MFCC------------|------------SSC------------| ')
    for label in range(0,len(labels)):
        print('==========')
        r,fpr,p,a,f1 = metrics(true_positive[label],false_positive[label], true_positive[label], false_negative[label])
        r_ssc,fpr_ssc,p_ssc,a_ssc,f1_ssc = metrics(true_positive_ssc[label],false_positive_ssc[label], true_positive_ssc[label], false_negative_ssc[label])
        str_label = transform(label)
        print('|{}| {} | {} | {} | {} | {} | {} | {} | {} |'.format(str_label,cust_round(fpr),cust_round(r),cust_round(a),cust_round(p),cust_round(fpr_ssc),cust_round(r_ssc),cust_round(a_ssc),cust_round(p_ssc)))
        
        
        R.append(r)
        A.append(a)
        P.append(p)
        F1.append(f1)
        FPR.append(fpr) 
        R_ssc.append(r_ssc)
        A_ssc.append(a_ssc)
        P_ssc.append(p_ssc)
        F1_ssc.append(f1_ssc)
        FPR_ssc.append(fpr_ssc)  


        
    # hist(R,R_ssc,labels,'Recall')
    # hist(FPR,FPR_ssc,labels,'False positive rate')
    # hist(A,A_ssc,labels,'Accuracy')
    # hist(P,P_ssc,labels,'Precision')
    # hist(F1,F1_ssc,labels,'F1 score')
        

        
    print('|================================================================|')     
    


    im1 = heatmap(np.array(matrix),labels,labels,ax=None,cmap='YlGn',cbarlabel="Confusion level MFCC")

    annotate_heatmap(im1,data = np.array(unchanged), valfmt="{x:d}", size=7, threshold=20)
    im2 = heatmap(np.array(matrix_ssc),labels,labels,ax=None,cmap='YlGn',cbarlabel="Confusion level SSC")
    annotate_heatmap(im2,data = np.array(unchanged_ssc), valfmt="{x:d}", size=7, threshold=20)
    break

    
