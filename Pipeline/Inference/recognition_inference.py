""" Preprocessing for page segmentation """

import random
from typing import Tuple

import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('Pipeline/Preprocessing')
from recognition_preprocessor import RecognitionPreprocessor
sys.path.append('Pipeline')
import model_settings as settings
sys.path.append('Pipeline/Inference')
from inference import Inference
sys.path.append('Pipeline/Postprocessing')
from spell_correction import correction
sys.path.append('Pipeline/utils')
from utils import read_charlist
from utils import simple_decode


class RecognitionInference(Inference):
    """ Class for inference word image """

    def __init__(self, 
                model_name: str = "ImprovedPen2Text_v7",
                char_list_path: str = settings.CHAR_DIR, 
                img_size: Tuple = (settings.HEIGHT, settings.WIDTH), 
                max_len: int = settings.MAX_LEN, 
                batch_size: int = settings.BATCH_SIZE) -> None:

        super().__init__(model_name, img_size, batch_size, "Recognition")
        self.char_list = read_charlist(char_list_path)
        self.preprocessor = RecognitionPreprocessor(
            img_size=img_size, char_list=self.char_list, max_len=max_len, batch_size=batch_size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predict word from image or array """
        result = []
        word_images = x
        processed_words = []
        for word in word_images:
            processed_word = self.preprocessor.process_inference(word)
            processed_words.append(processed_word)
            cv2.imshow("img", processed_word)
            cv2.waitKey(0)

        try:
            predictions = self.model.predict(processed_words, verbose=0)
        except:
            return []
        out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                                greedy=False)[0][0])

        for i, x in enumerate(out):
            label = simple_decode(x, self.char_list)
            if(label.isalpha()):
                label = correction(label)
            result.append(label)
        
        return result