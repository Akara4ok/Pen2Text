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


class PageSegPreprocessor(Preprocessor):
    """ Class for preprocessing for images """

    def __init__(self,
                 img_size: Tuple[int, int],
                 batch_size: int) -> None:

        super().__init__(img_size, batch_size)

    def process_single(self, path: tf.Tensor, bounding_boxes: tf.Tensor) -> Tuple:
        """ Create mask for single image """

        img = get_img(path.numpy().decode("utf-8"))
        mask = np.zeros_like(img)
        for x, y, w, h in bounding_boxes.numpy():
            mask[y:y+h, x:x+w] = 255

        min_x = 100000
        min_y = 100000
        max_x = -1
        max_y = -1
        
        for x, y, w, h in bounding_boxes.numpy():
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
        
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

        img = cv2.resize(img, self.img_size)
        img = img / 255
        img = np.expand_dims(img,axis=-1)

        mask = cv2.resize(mask, self.img_size)
        mask = mask / 255
        mask = np.expand_dims(mask,axis=-1)

        return (img, mask)
    
    def process_batch(self, batch: list) -> list:
        """ Create masks for whole batch """
        pass