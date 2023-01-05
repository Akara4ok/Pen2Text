""" Preprocessing for page segmentation """

import random
from typing import Tuple

import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('Pipeline/Preprocessing')
from page_seg_preprocessing import PageSegPreprocessor
sys.path.append('Pipeline')
import model_settings as settings
sys.path.append('Pipeline/Inference')
from inference import Inference
sys.path.append('Pipeline/utils')
from utils import read_charlist
from utils import simple_decode
from utils import custom_image_resize_sizes
sys.path.append('Pipeline/Inference/AStarSeg')
from astar_page_seg import AStarPageSegInference

class PageSegInference(Inference):
    """ Class for inference line segmentation to words """

    def __init__(self, 
                model_name: str = "PageSegUnet_latest",
                img_size: Tuple = (512, 512), 
                batch_size: int = settings.BATCH_SIZE) -> None:

        super().__init__(model_name, img_size, batch_size, "PageSeg")
        self.preprocessor = PageSegPreprocessor(batch_size=self.batch_size, img_size=self.img_size)
        self.astar_improvement = AStarPageSegInference()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predict words imgs from line """
        pages = x
        result = []
        processed_pages = []
        for page in pages:
            processed_page = self.preprocessor.process_inference(page)
            processed_pages.append(processed_page)
        processed_pages = np.array(processed_pages)
        tresholded_pages = self.preprocessor.treshold_array(pages)
        pred_masks = self.model.predict(processed_pages, verbose = 0)
        pred_masks = (pred_masks * 255).astype('uint8')

        for index, mask in enumerate(pred_masks):
            _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
            # cv2.imshow("page", processed_pages[index])
            # cv2.imshow("mask", mask)
            # cv2.waitKey(0)
                            # for subline in sublines:

            (p_h, p_w) = pages[index].shape
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
                # img_line = processed_pages[index, y:y+h,x:x+w]
                # img_line = np.squeeze(img_line, axis=-1)

                try:
                    sublines = self.astar_improvement.predict(np.expand_dims(img_line, axis=0))
                except:
                    sublines = [img_line]
                # cv2.imshow("subline", img_line)
                # cv2.waitKey(0)
                # for subline in sublines:
                #     cv2.imshow("subline", subline)
                #     cv2.waitKey(0)

                # result.append(img_line)
                result.extend(sublines)

        return result