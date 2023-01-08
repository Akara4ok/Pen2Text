""" Class for page seg postprocessing """
import sys
import tensorflow as tf
import numpy as np
import cv2
sys.path.append("Pipeline/Postprocessing")
from postprocessor import Postprocessor
sys.path.append('Pipeline/Postprocessing/PageSegPostprocessing/AStarSeg')
from astar_page_seg import AStarPageSegInference
sys.path.append('Pipeline/utils')
from utils import custom_image_resize_sizes

class PageSegPostprocessing(Postprocessor):
    """ Class for page seg postprocessing """
    def __init__(self) -> None:
        self.astar_improvement = AStarPageSegInference()
        
    def process(self, x: tf.Tensor, tresholded_pages: np.ndarray) -> np.ndarray:
        result = []
        pred_masks = (x * 255).astype('uint8')

        for index, mask in enumerate(pred_masks):
            _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

            (p_h, p_w) = tresholded_pages[index].shape
            new_width = new_height = None
            if(p_w > 512):
                (new_width, new_height) = custom_image_resize_sizes(p_h, p_w, new_width=512)
            else:
                new_width = p_w
            
            if(new_height is not None):
                if(new_height > 512):
                    (new_width, new_height) = custom_image_resize_sizes(new_height, new_width, new_height=512)
            else:
                new_height = p_h

            koef = p_w / new_width
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            line_coordinates = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                line_coordinates.append((x, y, w, h))
        
            line_coordinates.sort(key=lambda x:x[1])
            for line in line_coordinates:
                (x, y, w, h) = line
                x = int(x * koef)
                y = int(y * koef)
                w = int(w * koef)
                h = int(h * koef)

                img_line = tresholded_pages[index, y:y+h,x:x+w]
                try:
                    sublines = self.astar_improvement.predict(np.expand_dims(img_line, axis=0))
                except:
                    sublines = [img_line]
                result.extend(sublines)
        
        return result