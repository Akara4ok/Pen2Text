""" Preprocessing for page segmentation """

import random
from typing import Tuple

import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('Pipeline/Preprocessing')
from line_seg_preprocessing import LineSegPreprocessor
sys.path.append('Pipeline')
import model_settings as settings
sys.path.append('Pipeline/Inference')
from inference import Inference
sys.path.append('Pipeline/utils')
from utils import read_charlist
from utils import simple_decode
from utils import custom_image_resize_sizes

class LineSegInference(Inference):
    """ Class for inference line segmentation to words """

    def __init__(self, 
                model_name: str = "LineSegUnet_latest",
                img_size: Tuple = (512, 512), 
                batch_size: int = settings.BATCH_SIZE) -> None:

        super().__init__(model_name, img_size, batch_size, "LineSeg")
        self.preprocessor = LineSegPreprocessor(batch_size=self.batch_size, img_size=self.img_size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predict words imgs from line """
        lines = x
        result = []
        processed_lines = []
        for line in lines:
            # cv2.imshow("line", line)
            # cv2.waitKey(0)
            processed_line = self.preprocessor.process_inference(line)
            processed_lines.append(processed_line)

        processed_lines = np.array(processed_lines)
        pred_masks = self.model.predict(processed_lines, verbose = 0)
        pred_masks = (pred_masks * 255).astype('uint8')

        for index, mask in enumerate(pred_masks):
            _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
            if(lines[index].ndim < 2):
                continue
            (l_h, l_w) = lines[index].shape
            new_width = new_height = None
            if(l_w > 512):
                (new_width, new_height) = custom_image_resize_sizes(l_h, l_w, new_width=512)
            else:
                new_width = l_w
            
            if(new_height is not None):
                if(new_height > 512):
                    (new_width, new_height) = custom_image_resize_sizes(new_width, new_height, new_height=512)
            else:
                new_height = l_h

            if(new_width is None or new_width == 0):
                koef = 0
            else:
                koef = l_w / new_width

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            word_coordinates = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                word_coordinates.append((x, y, w, h))
        
            word_coordinates.sort(key=lambda x:x[0])
            for word in word_coordinates:
                (x, y, w, h) = word
                x = int(x * koef)
                y = int(y * koef)
                w = int(w * koef)
                h = int(h * koef)
                word_img = lines[index, y:y+h,x:x+w]
                if(word_img.shape[0] < 3 or word_img.shape[1] < 3):
                    continue
                white = np.sum(word_img)
                total = word_img.shape[0] * word_img.shape[1] 
                if(white / total < 0.03):
                    continue
                # cv2.imshow("img", word_img)
                # cv2.waitKey(0)
                result.append(word_img)
    
        return result
