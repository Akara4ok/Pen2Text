""" Abstract file for preprocessing """

""" Preprocessing for page segmentation """

import random
from typing import Tuple

import sys
import cv2
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod


class Postprocessor(ABC):
    """ Class for preprocessing for images """

    @abstractmethod
    def process(self, x: tf.Tensor) -> np.ndarray:
        """ Abstract method for processing single input """
        pass