"""Displays a batch of outputs after every epoch."""
import sys
import numpy as np
import tensorflow as tf
sys.path.append('Pipeline/utils')
from utils import decode_batch_predictions
import keras.backend as K
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
from jiwer import wer

class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, x, y, model, char_list):
        super().__init__()
        self.x = x
        self.y = y
        self.model = model
        self.char_list = char_list

    def on_epoch_end(self, epoch: int, logs=None):
        # predict outputs on validation images
        samples = []
        indexes = []
        for i in np.random.randint(0, len(self.x), 3):
            samples.append(self.x[i])
            indexes.append(i)
        samples = np.array(samples)
            
        prediction = self.model.predict(samples)
        
        # use CTC decoder
        out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                                greedy=False)[0][0])

        i = 0
        for x in out:
            print("original_text =  ", self.y[indexes[i]])
            print("predicted text = ", end = '')
            for p in x:  
                if int(p) != -1:
                    print(self.char_list[int(p)], end = '')       
            print('\n')
            i+=1

        # targets = self.y
        # samples = np.array(self.x)
        # prediction = self.model.predict(samples)
        # out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
        #                         greedy=False)[0][0])
        # decode_predictions = []
        

        # correct = 0
        # for (index, x) in enumerate(out):
        #     label = ''
        #     for p in x:  
        #         if int(p) != -1:
        #             label += self.char_list[int(p)]
        #     if(label == targets[index]):
        #         correct += 1
        #     else:
        #         print(label, targets[index])

        # wer_score = 1 - correct / len(targets)
        # print("-" * 100)
        # print(f"Word Error Rate: {wer_score:.4f}")
        # print("-" * 100)

        