""" PenTextService """
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.append('Pipeline')
from pipeline import Pipeline

class PenTextService():
    """ Main logic and validation before model pipeline """
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
    
    def process(self, files: list, language: str):
        """ Define file type and process it """
        images = []
        results = []
        language = language.upper()
        for file in files:
            if (file.content_type in ["image/jpeg", "image/png", "image/bmp"]):
                input_image = Image.open(file).convert("L")
                image = np.array(input_image)
                images.append(image)
        
        if(images):
            results.extend(self.pipeline.process_images(images, language))
        
        return results

    def process_images(self, images: list, language: str) -> list:
        """ Process images """
        results = self.pipeline.process_images(images, language)
        return results
    
    def process_pdf(self, pdfs: list, language: str) -> list:
        """ Process pdfs """
        pass