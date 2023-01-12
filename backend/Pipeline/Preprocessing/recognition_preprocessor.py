""" Preprocessing for images """

from email.policy import default
import random
from typing import Tuple

import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('Pipeline/Preprocessing')
from preprocessor import Preprocessor
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/utils')
from utils_types import Batch
from utils_types import Sample
from utils import get_img
from PIL import Image

class RecognitionPreprocessor(Preprocessor):
    """ Class for preprocessing for images """

    def __init__(self,
                 img_size: Tuple[int, int],
                 char_list: list,
                 max_len: int,
                 batch_size: int,
                 clustering_percent: float = 1,
                 data_augmentation: bool = False,
                 line_mode: bool = False) -> None:

        super().__init__(img_size, batch_size)
        self.data_augmentation = data_augmentation
        self.line_mode = line_mode
        self.clustering_percent = clustering_percent
        self.clustered_in_batch = 0
        self.char_list = char_list
        self.max_len = max_len

    def data_augment(self, img):
        """Apllying data augmentation."""
        res_image = img
        height, width = self.img_size
        current_height, current_width = res_image.shape
        # photometric data augmentation
        # TODO delete this part if accuracy will be low
        def gaussian_koef():
            return random.randint(1, 3) * 2 + 1
        if random.random() < 0.25:
            img = cv2.GaussianBlur(
                img, (gaussian_koef(), gaussian_koef()), 0)
        if random.random() < 0.25:
            img = cv2.dilate(img, np.ones((gaussian_koef(), gaussian_koef())))
        if random.random() < 0.25:
            img = cv2.erode(img, np.ones((gaussian_koef(), gaussian_koef())))

        # geometric data augmentation
        resized_koef = min(width / current_width, height / current_height)
        resized_koef_x = resized_koef * np.random.uniform(0.75, 1.05)
        resized_koef_y = resized_koef * np.random.uniform(0.75, 1.05)

        # random position around center
        low_xc = (width - current_width * resized_koef_x) / 2
        low_yc = (height - current_height * resized_koef_y) / 2
        clipped_xc = max((width - current_width * resized_koef_x) / 2, 0)
        clipped_yc = max((height - current_height * resized_koef_y) / 2, 0)
        xc_bias = low_xc + np.random.uniform(-clipped_xc, clipped_xc)
        yc_bias = low_yc + np.random.uniform(-clipped_yc, clipped_yc)

        color = 1

        transform_matrix = np.float32(
            [[resized_koef_x, 0, xc_bias], [0, resized_koef_y, yc_bias]])
        target = np.full(
            (self.img_size[0], self.img_size[1]), color, dtype=np.float32)

        res_image = cv2.warpAffine(res_image.astype(np.float32), transform_matrix,
                                    dsize=(
                                        self.img_size[1], self.img_size[0]),
                                    flags=cv2.INTER_AREA, dst=target,
                                    borderMode=cv2.BORDER_TRANSPARENT)
        _, res_image = cv2.threshold(res_image, 0.8, 1, cv2.THRESH_BINARY_INV)
        res_image = np.expand_dims(res_image,axis=-1)
        return res_image

    def process_img(self, img: np.ndarray, isInference: bool = False) -> np.ndarray:
        """Clusterring, resizing and apllying data augmentation."""
        if (img is None) or (img.shape[0] <= 1 or img.shape[1] <= 1) or (img.shape[0] / img.shape[1] < 0.05):
            img = np.zeros(self.img_size)

        res_image = img / 255 if img.dtype == np.uint8 else img

        # general preprocessing
        old_width = img.shape[1]
        old_height = img.shape[0]
        scale_percent = min(
            self.img_size[1] / old_width, self.img_size[0] / old_height)

        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        resized = cv2.resize(res_image.astype('float32'),
                                (width, height), interpolation=cv2.INTER_NEAREST)

        
        if(self.data_augmentation and not isInference):
            res_image = self.data_augment(res_image)
            return res_image

        color = 0 if isInference else 1
        new_height = self.img_size[0]
        new_width = self.img_size[1]
        old_height = resized.shape[0]
        old_width = resized.shape[1]
        pad_img = np.full(
            (self.img_size[0], self.img_size[1]), color, dtype=np.float32)

        # compute center offset
        y_center = max((new_height - old_height) // 2, 0)
        x_center = max((new_width - old_width) // 2, 0)

        pad_img[y_center:y_center+old_height,
                x_center:x_center+old_width] = resized
        res_image = pad_img

        if (not isInference):
            _, res_image = cv2.threshold(res_image, 0.8, 1, cv2.THRESH_BINARY_INV)
        else:
            _, res_image = cv2.threshold(res_image, 0.1, 1, cv2.THRESH_BINARY)
        
        # cv2.imshow("word", res_image * 255)
        # cv2.waitKey(0)
            
        res_image = np.expand_dims(res_image,axis=-1)

        return res_image

    def process_text(self, text):
        """ Process text """
        #print("---------", text, "-----------")
        processed_text = []
        for char in text:
            try:
                processed_text.append(self.char_list.index(char))
            except:
                pass
        #print("---------", processed_text, "-----------")
        return processed_text

    def process_single(self, x: tf.Tensor, y: tf.Tensor) -> Sample:
        """ Process single img and text to img """
        img = get_img(x.numpy().decode("utf-8"))
        res_img = self.process_img(img)

        res_text = self.process_text(y.numpy().decode("utf-8"))
        return (res_img, res_text)
    
    def process_inference(self, x: np.ndarray) -> Sample:
        """ Process single img for inference"""
        res_img = self.process_img(x, True)
        return res_img

    def process_batch(self, batch: Batch) -> Batch:
        """ Process batch of input"""
        self.clustered_in_batch = 0
        if self.line_mode:
            batch = self._create_text_line(batch)

        res_imgs = np.array([self.process_img(img)
                             for img in batch.imgs])

        res_texts = np.array([self.process_text(text)
                              for text in batch.texts])
        res_texts = tf.keras.utils.pad_sequences(
            res_texts, maxlen=self.max_len, padding='post', value=len(self.char_list))

        return Batch(res_imgs, res_texts)
