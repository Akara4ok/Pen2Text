""" Controllers for Pen2Text pipeline """

import sys
sys.path.append('Services')
from pen_text_service import PenTextService

sys.path.append('Exceptions')
from pipeline_exceptions import PipelineException
from file_exceptions import BadFileContentException
from file_exceptions import NotSupportContentTypeException
from language_exceptions import NotSupportLanguageException

class PenTextController():
    def __init__(self, pen_text_service: PenTextService):
        self.pen_text_service = pen_text_service

    def process(self, files: list, language: str = "English") -> tuple:
        try:
            plain_text = self.pen_text_service.process(files, language)
            return {
                "data": {
                    "plain_text": plain_text
                }
            }, 200

        except NotSupportLanguageException as language_ex:
            return {
                "errors": [{
                    "message": str(language_ex)
                }]
            }, language_ex.status

        except NotSupportContentTypeException as content_type_ex:
            return {
                "errors": [{
                    "message": str(content_type_ex)
                }]
            }, content_type_ex.status

        except BadFileContentException as content_ex:
            return {
                "errors": [{
                    "message": str(content_ex)
                }]
            }, content_ex.status

        except PipelineException as pipeline_ex:
            return {
                "errors": [{
                    "message": str(pipeline_ex)
                }]
            }, pipeline_ex.status

        except Exception as ex:
            return {
                "errors": [{
                    "message": str(ex)
                }]
            }, 500
