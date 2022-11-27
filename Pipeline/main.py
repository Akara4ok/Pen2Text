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
from line_seg_preprocessing import LineSegPreprocessor
from recognition_preprocessor import RecognitionPreprocessor
sys.path.append('Pipeline/Model')
from unet import UNet
sys.path.append('Pipeline/utils')
from utils import last_checkpoint
sys.path.append('Pipeline/Training/Callbacks')
from convert2onnx import ConvertCallback
sys.path.append('Pipeline/utils')
from utils import read_charlist
from utils import simple_decode
sys.path.append('Pipeline/Postprocessing')
from spell_correction import correction

data_loader = DataLoaderIAM(Path("Data/IAM Dataset"),
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)
char_list = read_charlist("./Pipeline/CharList.txt")
max_len = data_loader.get_max_len()

train, val, test = data_loader.split_for_page_segmentation(shuffle=False)
page_preprocessor = PageSegPreprocessor(batch_size=settings.BATCH_SIZE_SEG, img_size=(512, 512))
line_preprocessor = LineSegPreprocessor(batch_size=settings.BATCH_SIZE_SEG, img_size=(512, 512))
word_preprocessor = RecognitionPreprocessor(img_size=(settings.HEIGHT, settings.WIDTH), char_list=char_list, max_len=max_len, batch_size=settings.BATCH_SIZE)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train[0], tf.ragged.constant(train[1]))
    ).map(
            lambda x, y: tf.py_function(page_preprocessor.process_single, [x, y], [tf.float32, tf.float32]), 
            num_parallel_calls=tf.data.AUTOTUNE
            ).padded_batch(
                settings.BATCH_SIZE_SEG,
                padded_shapes=([None, None, 1], [None, None, 1])
                ).prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val[0], tf.ragged.constant(val[1]))
    ).map(
        lambda x, y: tf.py_function(page_preprocessor.process_single, [x, y], [tf.float32, tf.float32]), 
        num_parallel_calls=tf.data.AUTOTUNE
        ).padded_batch(
            settings.BATCH_SIZE_SEG, 
            padded_shapes=([None, None, 1], [None, None, 1])
            ).prefetch(buffer_size=tf.data.AUTOTUNE)

page_model_name = "PageSegUnet_v1"
page_model = tf.keras.models.load_model("./Models/PageSeg/Models/" + page_model_name + "/tf", compile=False)

line_model_name = "LineSegUnet_v1"
line_model = tf.keras.models.load_model("./Models/LineSeg/Models/" + line_model_name + "/tf", compile=False)

word_model_name = "ImprovedPen2Text_v7"
word_model = tf.keras.models.load_model("./Models/Recognition/Models/" + word_model_name + "/tf", compile=False)


for batch in train_dataset:
    for img, mask in zip(batch[0], batch[1]):
        img = img.numpy()
        mask = mask.numpy()
        preproc_img = np.expand_dims(img, axis=0)
        pred = page_model.predict(preproc_img)
        pred = (np.squeeze(pred, axis=0) * 255).astype('uint8')
        _, pred = cv2.threshold(pred, 252, 255, cv2.THRESH_BINARY)

        result = img
        contours, hier = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_coordinates = []
        margin = 10
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            line_coordinates.append((x, y, w, h))
            # result = cv2.rectangle(result, (x, y), (x+w,y+h), 255, 1)

        line_coordinates.sort(key=lambda x:x[1])
        words_imgs = []
        words = []

        for line in line_coordinates:
            # for line in line_coordinates:
            (x, y, w, h) = line
            img_line = np.squeeze(img[y:y+h,x:x+w], axis = 2)
            img_line_proc = line_preprocessor.process_inference(img_line)
            # print(img_line_proc.shape)
            pred_word = line_model.predict(np.expand_dims(img_line_proc, axis=0))
            pred_word = (np.squeeze(pred_word, axis=0) * 255).astype('uint8')
            _, pred_word = cv2.threshold(pred_word, 100, 255, cv2.THRESH_BINARY)
            result_words = np.squeeze(img_line_proc, axis = 2)
            # print('-------', result_words.shape, '----------')
            contours, hier = cv2.findContours(pred_word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            word_coordinates = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                word_coordinates.append((x, y, w, h))
                result_words = cv2.rectangle(result_words, (x, y), (x+w,y+h), 255, 1)
            
            word_coordinates.sort(key=lambda x:x[0])
            for word in word_coordinates:
                (x, y, w, h) = word
                word_img = img_line[y:y+h,x:x+w]
                processed_word = word_preprocessor.process_inference(word_img)

                words_imgs.append(processed_word)
                # cv2.imshow("img", processed_word)
                # cv2.waitKey(0)


        predictions = word_model.predict(words_imgs, verbose=0)
        out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                                greedy=False)[0][0])
        for i, x in enumerate(out):
            label = simple_decode(x, char_list)
            # print("-" * 100)
            # print("Original:", target)
            # print("Predicted:", label)
            if(label.isalpha()):
                label = correction(label)
                # print("Corrected:", label)
            words.append(label)
        
        print("'", ' '.join(words), "'")
        
        cv2.imshow("img", img)
        cv2.waitKey(0)