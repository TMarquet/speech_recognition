# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:50:32 2021

@author: martho
"""

import tensorflow as tf
import os
from tensorflow.keras.models import load_model
# Convert the model
from preprocessing import *
model = load_model(PATH_MODELS + 'model.h5')
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
tflite_model = converter.convert()

# Save the model.
with open(PATH_MODELS +'model.tflite', 'wb') as f:
  f.write(tflite_model)