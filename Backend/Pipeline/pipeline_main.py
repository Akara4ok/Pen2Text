""" Init inferences for Pipeline"""
import sys
sys.path.append('Pipeline/Inference')
from recognition_inference import RecognitionInference
from line_seg_inference import LineSegInference
from page_seg_inference import PageSegInference

def init_inferences() -> tuple:
    word_inferences = {
        "ENGLISH": RecognitionInference(),
        "UKRAINIAN": RecognitionInference()
    }
    line_inference = LineSegInference()
    page_inference = PageSegInference()
    return word_inferences, line_inference, page_inference
