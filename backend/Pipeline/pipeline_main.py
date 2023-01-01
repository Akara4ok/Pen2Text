""" Init inferences for Pipeline"""
import sys
sys.path.append('Pipeline/Inference')
from recognition_inference import RecognitionInference
from line_seg_inference import LineSegInference
from page_seg_inference import PageSegInference
sys.path.append('Pipeline/Inference/AStarSeg')
from astar_page_seg import AStarPageSegInference
from stat_line_seg import StatLineSegInference


sys.path.append('Pipeline')
import model_settings as settings

def init_inferences() -> tuple:
    word_inferences = {
        "ENGLISH": RecognitionInference(model_name="ImprovedPen2Text_v7"),
        "UKRAINIAN": RecognitionInference(model_name="UkrPen2Text_latest",
                                            char_list_path=settings.UKR_CHAR_DIR, 
                                            correction_file=settings.TEXT_CORRECTION_FILE_UKR)
    }
    line_inference = LineSegInference()
    page_inference = PageSegInference()
    return word_inferences, line_inference, page_inference
