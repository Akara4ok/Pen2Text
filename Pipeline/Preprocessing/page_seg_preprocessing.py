""" Preprocessing for page segmentation """

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
from utils import pad_or_resize

class PageSegPreprocessor(Preprocessor):
    """ Class for preprocessing images for segmentation """

    def __init__(self,
                 img_size: Tuple[int, int],
                 batch_size: int) -> None:

        super().__init__(img_size, batch_size)

    def process_img(self, img):
        """ Process img """
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 35)
        img = img / 255 if img.dtype == np.uint8 else img
        cv2.imshow("img", img)
        cv2.waitKey(0)
        # _, img = cv2.threshold(img, 0.2, 1, cv2.THRESH_BINARY_INV)
        img = pad_or_resize(img, 512, 512)
        img = cv2.resize(img, self.img_size)
        img = np.expand_dims(img,axis=-1)
        return img


    def create_mask(self, img, bounding_boxes):
        """ Create mask from bounding boxes """
        mask = np.zeros_like(img)
        mask = mask / 255 if mask.dtype == np.uint8 else mask
        for x, y, w, h in bounding_boxes:
            mask[y:y+h, x:x+w] = 1
        
        min_x = 100000
        min_y = 100000
        max_x = -1
        max_y = -1
        
        for x, y, w, h in bounding_boxes:
            if max_x < x + w:
                max_x = x + w
            if min_x > x:
                min_x = x
            if max_y < y + h:
                max_y = y + h
            if min_y > y:
                min_y = y
        
        max_x += random.Random(settings.RANDOM_SEED).randint(0, 200)
        min_x -= random.Random(settings.RANDOM_SEED).randint(0, 200) 
        min_y -= random.Random(settings.RANDOM_SEED).randint(0, 50)
        max_y += random.Random(settings.RANDOM_SEED).randint(0, 200)

        img = img[min_y:max_y, min_x:max_x]
        mask = mask[min_y:max_y, min_x:max_x]
        mask = pad_or_resize(mask, 512, 512)

        mask = cv2.resize(mask, self.img_size)
        mask = np.expand_dims(mask,axis=-1)
        return img, mask

    def process_single(self, x: tf.Tensor, y: tf.Tensor) -> Tuple:
        """ Create mask for single image """

        img = get_img(x.numpy().decode("utf-8"))
        img, mask = self.create_mask(img, y.numpy())
        img = self.process_img(img)

        return (img, mask)
    
    def process_inference(self, x: np.ndarray) -> np.ndarray:
        img = self.process_img(x)
        return img
    
    def process_batch(self, batch: list) -> list:
        """ Create masks for whole batch """
        pass