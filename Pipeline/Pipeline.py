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

class Pipeline():
    """ Pipeline for infering image """
    def __init__(self) -> None: 
        self.word_inference = RecognitionInference()
        self.line_inference = LineSegInference()
        self.page_inference = PageSegInference()

    def process_images(self, images: list) -> str:
        results = []
        for img in images:
            line_imgs = self.page_inference.predict(np.expand_dims(img, axis=0))
            words = []

            for line in line_imgs:
                words_imgs = self.line_inference.predict(np.expand_dims(line, axis=0))
                words.extend(self.word_inference.predict(words_imgs))
            
            results.append(' '.join(words))
        return results