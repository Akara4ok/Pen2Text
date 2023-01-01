""" Pipeline for training model for page segmentation to line """
import sys
from path import Path
import cv2
from jiwer import wer
from jiwer import cer


sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/')
import model_settings as settings
from pipeline_main import init_inferences
from pipeline import Pipeline
sys.path.append('Pipeline/utils')
from utils import crop_img

word_inferences, line_inference, page_inference = init_inferences()
pipeline = Pipeline(word_inferences, line_inference, page_inference)


path = "../../TestInference/my_examples/test4.jpg"
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
text = pipeline.process_images([img], "ENGLISH")
print(text)