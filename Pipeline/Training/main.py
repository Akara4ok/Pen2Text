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
from loss_functions import ctc_loss
sys.path.append('Pipeline/utils')
from utils import read_charlist
sys.path.append('Pipeline/Training/Callbacks')
from text_example import CallbackEval

#read dataset
data_loader = DataLoaderIAM(Path("Data/IAM Dataset"), 10,
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT)
train, val, test = data_loader.split()

#read chars
charlist = read_charlist(settings.CHAR_DIR)
char_to_num = tf.keras.layers.StringLookup(vocabulary=charlist, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

#organizing dataset
train_dataset = IAMSequence(train[0], train[1], settings.BATCH_SIZE, char_to_num, 'train')
val_dataset = IAMSequence(val[0], val[1], settings.BATCH_SIZE, char_to_num, 'val')
print(len(val_dataset))

#training model
model = Pen2Text(charlist)
model.compile(loss=ctc_loss, optimizer = 'adam')
epochs = 10

validation_callback = CallbackEval(val_dataset, model, num_to_char)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    callbacks=[validation_callback],
    epochs=epochs,
)