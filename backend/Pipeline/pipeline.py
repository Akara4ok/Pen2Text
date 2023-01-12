""" Pipeline for infering image """
import sys
from path import Path
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import cv2
import numpy as np
import os

sys.path.append('Pipeline/utils')
from utils import get_img
sys.path.append('Pipeline/Inference')
from recognition_inference import RecognitionInference
from line_seg_inference import LineSegInference
from page_seg_inference import PageSegInference
sys.path.append('Exceptions')
from pipeline_exceptions import PipelineException

class Pipeline():
    """ Pipeline for infering image """
    def __init__(self, word_inferences: dict, line_inference: LineSegInference, page_inference: PageSegInference) -> None: 
        self.word_inferences = word_inferences
        self.line_inference = line_inference
        self.page_inference = page_inference

    def process_images(self, images: list, model_name) -> str:
        """ Process image pipeline """
        results = []
        
        try:
            for index, img in enumerate(images):
                line_imgs = self.page_inference.predict(np.expand_dims(img, axis=0))
                words = []

                for line in line_imgs:
                    words_imgs = self.line_inference.predict(np.expand_dims(line, axis=0))
                    words.extend(self.word_inferences[model_name].predict(words_imgs))
                
                results.append(' '.join(words))
        except Exception as ex:
            print(str(ex))
            raise PipelineException(index=index)

        return results