"""Class for manipulating data during training and infering"""

import abc
import sys
import cv2

sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/utils')
from utils_types import Batch
sys.path.append('Pipeline/Preprocessing')
from preprocessor import Preprocessor
import tensorflow as tf
import numpy as np


class IAMSequence(tf.keras.utils.Sequence):
    """Abstract class for manipulating IAM dataset"""
    def __init__(self, x_set: np.ndarray, y_set: np.ndarray, batch_size: int, char_to_num, mode: str):
        self.i = 0
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.curr_idx = 0
        self.mode = mode
        self.preprocessor = Preprocessor(img_size=(settings.HEIGHT, settings.WIDTH), char_to_num=char_to_num, clustering_percent=1)
        if(self.mode == 'train'):
            self.preprocessor = Preprocessor((settings.HEIGHT, settings.WIDTH), 
                                              char_to_num=char_to_num,
                                              clustering_percent=settings.CLUSTERING_PERCENT, 
                                              data_augmentation=True, 
                                              line_mode=settings.LINE_MODE)


    @abc.abstractmethod
    def __len__(self) -> int:
        """ Get len of dataset """
        return ((len(self.x_set) - 1) // self.batch_size) + 1


    def _get_img(self, i: int, mode: str = '') -> np.ndarray:
        if mode == '':
            img = cv2.imread(self.x_set[i], cv2.IMREAD_GRAYSCALE)
        return img

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Batch:
        """ Get next element of dataset """
        batch_range = range(idx, min(
            idx + self.batch_size, len(self.x_set)))

        imgs = np.array([self._get_img(i) for i in batch_range])
        texts = np.array([self.y_set[i] for i in batch_range])

        self.curr_idx += self.batch_size
        batch = Batch(imgs, texts)
        batch = self.preprocessor.process_batch(batch)
        self.i += 1
        # print("---------", self.i, "---------")
        # print(batch_range)
        # print(batch)
        # print('-------------------batch--------------')
        return batch

    def on_epoch_end(self) -> None:
        """ Updates indexes after each epoch """
        self.curr_idx = 0