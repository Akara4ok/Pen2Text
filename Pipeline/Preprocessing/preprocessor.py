""" Abstract file for preprocessing """

""" Preprocessing for page segmentation """

import random
from typing import Tuple

import sys
import cv2
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """ Class for preprocessing for images """

    def __init__(self,
                 img_size: Tuple[int, int],
                 batch_size: int) -> None:

        self.img_size = img_size
        self.batch_size = batch_size

    @abstractmethod
    def process_single(self) -> Tuple:
        """ Abstract method for processing single input """
        pass
    
    @abstractmethod
    def process_batch(self, batch: list) -> list:
        """ Abstract method for processing single batch """
        pass