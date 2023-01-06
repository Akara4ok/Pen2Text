""" PenTextService """
import sys
import io
import cv2
import numpy as np
from PIL import Image, ImageOps
import fitz

sys.path.append('Exceptions')
from language_exceptions import NotSupportLanguageException
from file_exceptions import NotSupportContentTypeException
from file_exceptions import BadFileContentException
from pipeline_exceptions import PipelineException
from network_exceptions import NotSupportNetworkException

sys.path.append('Pipeline')
from pipeline import Pipeline

sys.path.append('utils')
from file_handler import FileHandler

class PenTextService():
    """ Main logic and validation before model pipeline """
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.file_handler = FileHandler()
    
    def process(self, files: list, language: str, network_name: str) -> np.ndarray:
        """ Define file type and process it """
        language = language.upper()
        sanitize_network_name = ''
        if(language != "ENGLISH" and language != "UKRAINIAN"):
            raise NotSupportLanguageException()
        
        network_name = network_name.upper()
        if(network_name == "LETTERS"):
            pass
        elif (network_name == "LETTERS+NUMBERS"):
            network_name = "LETTERS_NUMBERS"
        elif (network_name == "ALL CHARS"):
            network_name = "ALL"
        else:
            raise NotSupportNetworkException()

        images, filenames = self.file_handler.processFiles(files)
        print(filenames)
        print(len(images))
        results = []
        
        model_name = language + "_" + network_name
        if(images):
            try:
                results.extend(self.pipeline.process_images(images, model_name))
            except PipelineException as pipeline_ex:
                index = pipeline_ex.index
                raise PipelineException(index=index, filename=filenames[index])
        
        return results