""" Class for line seg postprocessing """
import sys
import tensorflow as tf
import numpy as np
import cv2
sys.path.append("Pipeline/Postprocessing")
from postprocessor import Postprocessor
sys.path.append('Pipeline/utils')
from utils import custom_image_resize_sizes

class LineSegPostprocessing(Postprocessor):
    """ Class for line seg postprocessing """
    def process(self, x: tf.Tensor, thresholded_lines) -> np.ndarray:
        """ Postprocess segmentation results """
        result = []
        pred_masks = (x * 255).astype('uint8')

        for index, mask in enumerate(pred_masks):
            _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
            if(thresholded_lines[index].ndim < 2):
                continue
            (l_h, l_w) = thresholded_lines[index].shape
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
                word_img = thresholded_lines[index, y:y+h,x:x+w]
                if(word_img.shape[0] < 3 or word_img.shape[1] < 3):
                    continue
                white = np.sum(word_img)
                total = word_img.shape[0] * word_img.shape[1] 
                if(white / total < 0.03):
                    continue
                result.append(word_img)
    
        return result