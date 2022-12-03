""" Controllers for Pen2Text pipeline """

import sys
sys.path.append('Services')
from PenTextService import PenTextService

class PenTextController():
    def __init__(self, pen_text_service: PenTextService):
        self.pen_text_service = pen_text_service
    
    def process(self, files: list, language: str = "English") -> tuple:
        plain_text = self.pen_text_service.process(files, language)
        return {
            "plain_text": plain_text
        }, 200