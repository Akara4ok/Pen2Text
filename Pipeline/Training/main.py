""" Pipeline for training model """
import sys
import time
from path import Path
import cv2
import tensorflow as tf

sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/Dataset/IAM Dataset/')
from iam_sequence import IAMSequence
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/Model')
from pen2text import Pen2Text
from improved_ocr import ImprovedPen2Text
from loss_functions import ctc_loss
sys.path.append('Pipeline/utils')
from utils import read_charlist
sys.path.append('Pipeline/Training/Callbacks')
from text_example import CallbackEval
sys.path.append('Pipeline/Preprocessing')
from preprocessor import Preprocessor
import numpy as np


import fnmatch
import cv2
import numpy as np
import string
import time

from keras.utils import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle

#read dataset
data_loader = DataLoaderIAM(Path("Data/IAM Dataset"), 10,
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)
train, val, test = data_loader.split()

char_list = read_charlist("./Pipeline/CharList.txt")


training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []
 
#lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []
 
max_label_len = 0

preprocessor = Preprocessor(img_size=(settings.HEIGHT, settings.WIDTH), char_to_num=None, clustering_percent=1)
model=ImprovedPen2Text(char_list)

for i in range(len(train[0])):
        img = preprocessor.get_img(train[0][i])
        img=preprocessor.process_img2(img,(128, 32))
        img=np.expand_dims(img,axis=-1)
        # img = img/255.

        txt = train[1][i]
        
        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)
            
        orig_txt.append(txt)   
        train_label_length.append(len(txt))
        train_input_length.append(31)
        training_img.append(img)
        training_txt.append(preprocessor.encode_to_labels(txt, char_list)) 

for i in range(len(val[0])):
        img = preprocessor.get_img(val[0][i])
        img = preprocessor.process_img2(img, (128, 32))
        img = np.expand_dims(img,axis=-1)
        txt = val[1][i]
        
        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)
            
        valid_orig_txt.append(txt)   
        valid_label_length.append(len(txt))
        valid_input_length.append(31)
        valid_img.append(img)
        valid_txt.append(preprocessor.encode_to_labels(txt, char_list))

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = len(char_list))

 
filepath="/home/vlad/Projects/Pen2Text/Checkpoints/Test/cp-{epoch:04d}.ckpt"
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss', 
                             verbose=1,
                             save_weights_only = True, 
                             save_best_only=True, 
                             mode='auto')

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)


#model = ImprovedPen2Text(char_list)
model.compile(loss=ctc_loss, optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
validation_callback = CallbackEval(valid_img, valid_orig_txt, model, char_list)
callbacks_list = [checkpoint, validation_callback]

batch_size = settings.BATCH_SIZE
epochs = 30
model.fit(
    x=training_img, 
    y=train_padded_txt,
    batch_size=batch_size, 
    epochs = epochs, 
    validation_data = (valid_img, valid_padded_txt),
    verbose = 1, 
    callbacks = callbacks_list)
