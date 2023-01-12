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
sys.path.append('Pipeline/Postprocessing/PageSegPostprocessing')
from page_seg_postprocessing import PageSegPostprocessing

class PageSegInference(Inference):
    """ Class for inference line segmentation to words """

    def __init__(self, 
                model_name: str = "PageSegUnet_latest",
                img_size: Tuple = (512, 512), 
                batch_size: int = settings.BATCH_SIZE) -> None:

        super().__init__(model_name, img_size, batch_size, "PageSeg")
        self.preprocessor = PageSegPreprocessor(batch_size=self.batch_size, img_size=self.img_size)
        self.post_processor = PageSegPostprocessing()

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
        
        result = self.post_processor.process(pred_masks, tresholded_pages)

        return result