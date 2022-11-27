""" Inference """

""" Pipeline for training model for page segmentation to line """
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


directory = "Data/TestInference"
images = []

for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    images.append(get_img(path))

word_inference = RecognitionInference()
line_inference = LineSegInference()
page_inference = PageSegInference()

for img in images:
    line_imgs = page_inference.predict(np.expand_dims(img, axis=0))
    words = []

    for line in line_imgs:
        words_imgs = line_inference.predict(np.expand_dims(line, axis=0))
        words.extend(word_inference.predict(words_imgs))
    
    print("'", ' '.join(words), "'")
    
    cv2.imshow("img", img)
    cv2.waitKey(0)