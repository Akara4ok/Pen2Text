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


class LineSegPreprocessor(Preprocessor):
    """ Class for preprocessing for images """

    def __init__(self,
                 img_size: Tuple[int, int],
                 batch_size: int) -> None:

        super().__init__(img_size, batch_size)

    def process_img(self, img_line):     
        (height, width) = img_line.shape[:2]


        if(width > 512):
            img_line = custom_image_resize(img_line, width=512)
        else:
            to_pad = np.zeros((height,512-width))
            img_line = np.concatenate((img_line,to_pad),axis=1)
        
        (height, width) = img_line.shape[:2]

        if(height > 512):
            img_line = custom_image_resize(img_line, height=512)
        else:
            to_pad=np.zeros((512-height,width))
            img_line=np.concatenate((img_line,to_pad), axis=0)
        
        (height, width) = img_line.shape[:2]

        img_line = cv2.resize(img_line, self.img_size)
        img_line = np.expand_dims(img_line,axis=-1)

        return img_line

    def process_single(self, path: tf.Tensor, line_box: tf.Tensor, bounding_boxes: tf.Tensor) -> Tuple:
        """ Create mask for single line """

        img = get_img(path.numpy().decode("utf-8"))
        (x_line, y_line, width_line, height_line) = line_box.numpy()
        img_line = img[y_line:y_line+height_line, x_line:x_line+width_line]
        mask = np.zeros_like(img)
        for x, y, w, h in bounding_boxes.numpy():
            mask[y:y+h, x:x+w] = 255

        mask = mask[y_line:y_line+height_line, x_line:x_line+width_line]

        _, img_line = cv2.threshold(img_line, 150, 255, cv2.THRESH_BINARY_INV)

        
        (height, width) = img_line.shape[:2]

        if(width > 512):
            img_line = custom_image_resize(img_line, width=512)
            mask = custom_image_resize(mask, width=512)
        else:
            to_pad = np.zeros((height,512-width))
            img_line = np.concatenate((img_line,to_pad),axis=1)
            mask = np.concatenate((mask,to_pad),axis=1)
        
        (height, width) = img_line.shape[:2]

        if(height > 512):
            img_line = custom_image_resize(img_line, height=512)
            mask = custom_image_resize(mask, height=512)
        else:
            to_pad=np.zeros((512-height,width))
            img_line=np.concatenate((img_line,to_pad), axis=0)
            mask = np.concatenate((mask,to_pad), axis=0)
        
        (height, width) = img_line.shape[:2]

        img_line = cv2.resize(img_line, self.img_size)
        img_line = img_line / 255
        img_line = np.expand_dims(img_line,axis=-1)

        mask = cv2.resize(mask, self.img_size)
        mask = mask / 255
        mask = np.expand_dims(mask,axis=-1)

        return img_line, mask
    
    def process_batch(self, batch: list) -> list:
        """ Create masks for whole batch """
        pass