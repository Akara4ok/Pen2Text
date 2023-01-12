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
sys.path.append('Pipeline/Postprocessing/LineSegPostprocessing')
from line_seg_postprocessing import LineSegPostprocessing

class LineSegInference(Inference):
    """ Class for inference line segmentation to words """

    def __init__(self, 
                model_name: str = "LineSegUnet_latest",
                img_size: Tuple = (512, 512), 
                batch_size: int = settings.BATCH_SIZE) -> None:

        super().__init__(model_name, img_size, batch_size, "LineSeg")
        self.preprocessor = LineSegPreprocessor(batch_size=self.batch_size, img_size=self.img_size)
        self.post_processor = LineSegPostprocessing()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predict words imgs from line """
        lines = x
        processed_lines = []
        for line in lines:
            # cv2.imshow("line", line)
            # cv2.waitKey(0)
            processed_line = self.preprocessor.process_inference(line)
            processed_lines.append(processed_line)

        processed_lines = np.array(processed_lines)
        pred_masks = self.model.predict(processed_lines, verbose = 0)
        
        results = self.post_processor.process(pred_masks, lines)

        return results
