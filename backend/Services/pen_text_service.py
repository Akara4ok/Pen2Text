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

class PenTextService():
    """ Main logic and validation before model pipeline """
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.allowed_content_types = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "application/pdf"}
    
    def process(self, files: list, language: str, network_name: str):
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

        images = []
        filenames = []
        results = []
        for file in files:
            if (file.content_type not in self.allowed_content_types):
                raise NotSupportContentTypeException(file.filename, self.allowed_content_types)

            if(file.content_type == "application/pdf"):
                try:
                    pdf_images = self.flatten_pdf(file.read())
                except:
                    raise BadFileContentException(file.filename)

                images.extend(pdf_images)
                for _ in range(len(pdf_images)):
                    filenames.append(file.filename)
                continue

            try:
                input_image = Image.open(file).convert("L")
                input_image = ImageOps.exif_transpose(input_image)
            except:
                raise BadFileContentException(file.filename)
                
            image = np.array(input_image)
            images.append(image)
            filenames.append(file.filename)
        
        model_name = language + "_" + network_name
        if(images):
            try:
                results.extend(self.pipeline.process_images(images, model_name))
            except PipelineException as pipeline_ex:
                index = pipeline_ex.index
                raise PipelineException(index=index, filename=filenames[index])
        
        return results
    
    def flatten_pdf(self, pdf_name):
        """ Extract images from pdf """
        result = []
        doc = fitz.open(stream=pdf_name, filetype="pdf")
        for page in doc:
            for img in page.get_images():
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                # load it to PIL
                pil_image = Image.open(io.BytesIO(image_bytes))
                cv_image = np.array(pil_image)
                gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                result.append(gray_image)
        return result