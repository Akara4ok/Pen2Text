""" Pipeline for training model for page segmentation to line """
import sys
from path import Path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np

sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/Preprocessing')
from line_seg_preprocessing import LineSegPreprocessor
sys.path.append('Pipeline/Model')
from unet import UNet
sys.path.append('Pipeline/utils')
from utils import last_checkpoint
sys.path.append('Pipeline/Training/Callbacks')
from convert2onnx import ConvertCallback


data_loader = DataLoaderIAM(Path("Data/IAM Dataset"),
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)

train, val, test = data_loader.split_for_line_segmentation(shuffle=False)
preprocessor = LineSegPreprocessor(batch_size=settings.BATCH_SIZE_SEG, img_size=(512, 512))

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train[0], tf.ragged.constant(train[1]), tf.ragged.constant(train[2]))
    ).shuffle(
        settings.IMG_NUM,
        reshuffle_each_iteration=True
        ).map(
            lambda x, y, z: tf.py_function(preprocessor.process_single, [x, y, z], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.AUTOTUNE
            ).padded_batch(
                settings.BATCH_SIZE_SEG,
                padded_shapes=([None, None, 1], [None, None, 1])
                ).prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val[0], tf.ragged.constant(val[1]), tf.ragged.constant(val[2]))
    ).map(
        lambda x, y, z: tf.py_function(preprocessor.process_single, [x, y, z], [tf.float32, tf.float32]), 
        num_parallel_calls=tf.data.AUTOTUNE
        ).padded_batch(
            settings.BATCH_SIZE_SEG, 
            padded_shapes=([None, None, 1], [None, None, 1])
            ).prefetch(buffer_size=tf.data.AUTOTUNE)

model_name = "LineSegUnet_v1"
model = tf.keras.models.load_model("./Models/LineSeg/Models/" + model_name + "/tf", compile=False)


for batch in val_dataset:
    for img, mask in zip(batch[0], batch[1]):
        img = img.numpy()
        mask = mask.numpy()
        preproc_img = np.expand_dims(img, axis=0)
        pred = model.predict(preproc_img)
        pred = (np.squeeze(pred, axis=0) * 255).astype('uint8')
        _, pred = cv2.threshold(pred, 10, 255, cv2.THRESH_BINARY)

        result = img
        contours, hier = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a white rectangle to visualize the bounding rect
            result = cv2.rectangle(result, (x, y), (x+w,y+h), 255, 1)
            # coordinates.append([x,y,(x+w),(y+h)])

        cv2.imshow("img", img)
        cv2.imshow("mask", mask)
        cv2.imshow("pred", pred)
        cv2.imshow("result", result)
        cv2.waitKey(0)
    # break