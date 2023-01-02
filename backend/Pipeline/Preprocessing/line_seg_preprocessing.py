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
from utils import custom_image_resize
from utils import pad_or_resize


class LineSegPreprocessor(Preprocessor):
    """ Class for preprocessing for images """

    def __init__(self,
                 img_size: Tuple[int, int],
                 batch_size: int) -> None:

        super().__init__(img_size, batch_size)

    def process_img(self, img_line, isInference = False):
        """ Process image """
        # img_line = cv2.adaptiveThreshold(img_line, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 35)
        img_line = img_line / 255 if img_line.dtype == np.uint8 else img_line
        if (not isInference):
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 35)

        img_line = pad_or_resize(img_line, 512, 512)

        img_line = cv2.resize(img_line, self.img_size)

        img_line = np.expand_dims(img_line,axis=-1)

        return img_line

    def create_mask(self, img, line_box: np.ndarray, bounding_boxes: np.ndarray):
        """ Create mask for line segmentation """
        (x_line, y_line, width_line, height_line) = line_box
        img_line = img[y_line:y_line+height_line, x_line:x_line+width_line]
        mask = np.zeros_like(img)
        mask = mask / 255 if mask.dtype == np.uint8 else mask
        for x, y, w, h in bounding_boxes:
            mask[y:y+h, x:x+w] = 1

        mask = mask[y_line:y_line+height_line, x_line:x_line+width_line]
        mask = pad_or_resize(mask, 512, 512)
        mask = cv2.resize(mask, self.img_size)
        mask = np.expand_dims(mask,axis=-1)
        return img_line, mask

    def process_single(self, x: tf.Tensor, y: tf.Tensor) -> Tuple:
        """ Process single for line seg training """
        img = get_img(x.numpy().decode("utf-8"))
        line_box, bounding_boxes = y.numpy()
        line_box = np.concatenate(line_box)
        img_line, mask = self.create_mask(img, line_box, bounding_boxes)
        img_line = self.process_img(img_line)

        return img_line, mask
    
    def process_inference(self, x: np.ndarray) -> np.ndarray:
        """ Process single for line seg inference """
        img_line = self.process_img(x, True)
        return img_line
    
    def process_batch(self, batch: list) -> list:
        """ Create masks for whole batch """
        pass