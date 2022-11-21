""" Inference """

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
from page_seg_preprocessing import PageSegPreprocessor
sys.path.append('Pipeline/Model')
from unet import UNet
sys.path.append('Pipeline/utils')
from utils import last_checkpoint
sys.path.append('Pipeline/Training/Callbacks')
from convert2onnx import ConvertCallback


data_loader = DataLoaderIAM(Path("Data/IAM Dataset"),
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)

# train, val, test = data_loader.split_for_page_segmentation(shuffle=False)
# preprocessor = PageSegPreprocessor(batch_size=settings.BATCH_SIZE_SEG, img_size=(512, 512))

# train_dataset = tf.data.Dataset.from_tensor_slices(
#     (train[0], tf.ragged.constant(train[1]))
#     ).shuffle(
#         settings.IMG_NUM,
#         reshuffle_each_iteration=True
#         ).map(
#             lambda x, y: tf.py_function(preprocessor.process_single, [x, y], [tf.float32, tf.float32]), 
#             num_parallel_calls=tf.data.AUTOTUNE
#             ).padded_batch(
#                 settings.BATCH_SIZE_SEG,
#                 padded_shapes=([None, None, 1], [None, None, 1])
#                 ).prefetch(buffer_size=tf.data.AUTOTUNE)

# val_dataset = tf.data.Dataset.from_tensor_slices(
#     (val[0], tf.ragged.constant(val[1]))
#     ).map(
#         lambda x, y: tf.py_function(preprocessor.process_single, [x, y], [tf.float32, tf.float32]), 
#         num_parallel_calls=tf.data.AUTOTUNE
#         ).padded_batch(
#             settings.BATCH_SIZE_SEG, 
#             padded_shapes=([None, None, 1], [None, None, 1])
#             ).prefetch(buffer_size=tf.data.AUTOTUNE)

# model_name = "PageSegUnet_v1"
# model = tf.keras.models.load_model("./Models/PageSeg/Models/" + model_name + "/tf", compile=False)


# for batch in val_dataset:
#     for img, mask in zip(batch[0], batch[1]):
#         img = img.numpy()
#         mask = mask.numpy()
#         preproc_img = np.expand_dims(img, axis=0)
#         pred = model.predict(preproc_img)
#         pred = (np.squeeze(pred, axis=0) * 255).astype('uint8')
#         _, pred = cv2.threshold(pred, 252, 255, cv2.THRESH_BINARY)

#         result = img
#         contours, hier = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         line_coordinates = []
#         for c in contours:
#             x, y, w, h = cv2.boundingRect(c)
#             line_coordinates.append((x, y, w, h))
#             result = cv2.rectangle(result, (x, y), (x+w,y+h), 255, 1)

#         line_coordinates.sort(key=lambda x:x[1])

#         cv2.imshow("img", img)
#         cv2.imshow("mask", mask)
#         cv2.imshow("pred", pred)
#         cv2.imshow("result", result)
#         cv2.waitKey(0)