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
max_len = data_loader.get_max_len()

char_list = read_charlist("./Pipeline/CharList.txt")

train_dataset = IAMSequence(train[0], train[1], settings.BATCH_SIZE, char_list, max_len, 'train')
val_dataset = IAMSequence(val[0], val[1], settings.BATCH_SIZE, char_list, max_len, 'val')

filepath="/home/vlad/Projects/Pen2Text/Checkpoints/Test/cp-{epoch:04d}.ckpt"
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss', 
                             verbose=1,
                             save_weights_only = True, 
                             save_best_only=True, 
                             mode='auto')

callbacks_list=[checkpoint]


model=ImprovedPen2Text(char_list)
model.compile(loss=ctc_loss, optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))

value = train_dataset[0]
epochs = 30
model.fit(
    train_dataset,
    epochs = epochs, 
    validation_data = val_dataset,
    verbose = 1, 
    callbacks = callbacks_list)
