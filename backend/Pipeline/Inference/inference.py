"""Abstract class for inference"""

from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class Inference(ABC):
    """Abstract class for inference"""

    def __init__(self,
                 model_name: str,
                 img_size: Tuple,
                 batch_size: int,
                 type: str):

        self.model_name = model_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = tf.keras.models.load_model("./Models/" + type + "/Models/" + model_name + "/tf", compile=False)

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Inference some x """
        pass
