""" Init controllers """

import sys
sys.path.append('Controllers')
from pen_text_controller import PenTextController
sys.path.append('Services')
from pen_text_service import PenTextService

def init_controllers(pen_text_service: PenTextService) -> PenTextController:
    """ Init pen_text_controller """
    pen_text_controller = PenTextController(pen_text_service)
    return pen_text_controller