""" Pipeline for training model for page segmentation to line """
import sys
from path import Path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/Preprocessing')
from page_seg_preprocessing import PageSegPreprocessor
sys.path.append('Pipeline/Model')
from unet import UNet
sys.path.append('Pipeline/utils')
from utils import last_checkpoint
sys.path.append('Pipeline/Training/Callbacks')
from convert2onnx import ConvertCallback


data_loader = DataLoaderIAM(Path("Data/IAM Dataset"),
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)

train, val, test = data_loader.split_for_page_segmentation(shuffle=False)
preprocessor = PageSegPreprocessor(batch_size=settings.BATCH_SIZE_SEG, img_size=(512, 512))

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train[0], tf.ragged.constant(train[1]))
    ).shuffle(
        settings.IMG_NUM,
        reshuffle_each_iteration=True
        ).map(
            lambda x, y: tf.py_function(preprocessor.process_single, [x, y], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.AUTOTUNE
            ).padded_batch(
                settings.BATCH_SIZE_SEG,
                padded_shapes=([None, None, 1], [None, None, 1])
                ).prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val[0], tf.ragged.constant(val[1]))
    ).map(
        lambda x, y: tf.py_function(preprocessor.process_single, [x, y], [tf.float32, tf.float32]), 
        num_parallel_calls=tf.data.AUTOTUNE
        ).padded_batch(
            settings.BATCH_SIZE, 
            padded_shapes=([None, None, 1], [None, None, 1])
            ).prefetch(buffer_size=tf.data.AUTOTUNE)

model_name = "PageSegUnet_v1"

model=UNet()
model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'], optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))

checkpoint_dir = "./Models/PageSeg/Checkpoints/" + model_name + "/"

loadFromCheckpoint = False
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

tf_path = "./Models/PageSeg/Models/" + model_name + "/tf"
fullModelSave = ModelCheckpoint(filepath=tf_path, 
                             monitor='val_loss', 
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
onnx_path = "./Models/PageSeg/Models/" + model_name + "/onnx/model.onnx"
convert = ConvertCallback(tf_path, onnx_path)

log_dir = "./Models/PageSeg/Logs/" + model_name + "/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

callbacks_list = [checkpoint, tensorboard_callback, fullModelSave, convert]

epochs = 5
model.fit(
    train_dataset,
    epochs = epochs, 
    validation_data = val_dataset,
    verbose = 1,
    initial_epoch=init_epoch,
    callbacks = callbacks_list)