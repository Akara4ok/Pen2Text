""" Init services """

import sys
sys.path.append('Pipeline')
from pipeline import Pipeline
sys.path.append('Services')
from PenTextService import PenTextService

def init_services(pipeline: Pipeline) -> PenTextService:
    pen_text_service = PenTextService(pipeline)
    return pen_text_service