

""" Pipeline for training model """
import sys
from path import Path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

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
from utils import last_checkpoint
sys.path.append('Pipeline/Training/Callbacks')
from text_example import CallbackEval
from convert2onnx import ConvertCallback
sys.path.append('Pipeline/Preprocessing')
from preprocessor import Preprocessor


#read dataset
data_loader = DataLoaderIAM(Path("Data/IAM Dataset"),
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)
train, val, test = data_loader.split()

char_list = read_charlist("./Pipeline/CharList.txt")
max_len = settings.MAX_LEN


preprocessor = Preprocessor(img_size=(settings.HEIGHT, settings.WIDTH), char_list=char_list, max_len=max_len, batch_size=settings.BATCH_SIZE)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train[0], train[1])
    ).map(
        lambda x, y: tf.py_function(preprocessor.process_single, [x, y], [tf.float32, tf.uint8]), 
        num_parallel_calls=tf.data.AUTOTUNE
        ).padded_batch(
            settings.BATCH_SIZE, 
            padded_shapes=([None, None, 1], [None]),
            padding_values=(0., tf.cast(len(char_list), dtype=tf.uint8))
            ).shuffle(
                settings.IMG_NUM,
                reshuffle_each_iteration=True
                ).prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val[0], val[1])
    ).map(
        lambda x, y: tf.py_function(preprocessor.process_single, [x, y], [tf.float32, tf.uint8]), 
        num_parallel_calls=tf.data.AUTOTUNE
        ).padded_batch(
            settings.BATCH_SIZE, 
            padded_shapes=([None, None, 1], [None]),
            padding_values=(0., tf.cast(len(char_list), dtype=tf.uint8))
            ).prefetch(buffer_size=tf.data.AUTOTUNE)


model=ImprovedPen2Text(char_list)
model.compile(loss=ctc_loss, optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))

checkpoint_dir = "./Checkpoints/Test/"
loadFromCheckpoint = True
init_epoch = 0
if(loadFromCheckpoint):
    init_epoch, checkpoint_name = last_checkpoint(checkpoint_dir)
    model.load_weights(checkpoint_dir + checkpoint_name)


checkpoint_path = checkpoint_dir + "cp-{epoch:04d}.ckpt"
checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             monitor='val_loss', 
                             verbose=1,
                             save_weights_only = True, 
                             mode='auto')

tf_path = "./Models/Test/tf"
fullModelSave = ModelCheckpoint(filepath=tf_path, 
                             monitor='val_loss', 
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
onnx_path = "./Models/Test/onnx/model.onnx"
convert = ConvertCallback(tf_path, onnx_path)

validation_callback = CallbackEval(val_dataset, model, char_list)
callbacks_list = [checkpoint, fullModelSave, convert, validation_callback]

epochs = 30
model.fit(
    train_dataset,
    epochs = epochs, 
    validation_data = val_dataset,
    verbose = 1,
    initial_epoch=init_epoch,
    callbacks = callbacks_list)