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

data_loader = DataLoaderIAM(Path("Data/IAM Dataset"),
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)

train, val, test = data_loader.split_for_page_segmentation(shuffle=False)
train_text, val_text, test_text = data_loader.get_text(shuffle=False)

word_inferences, line_inference, page_inference = init_inferences()
pipeline = Pipeline(word_inferences, line_inference, page_inference)

i = 0
wer_history = []
cer_history = []
for (path, boxes, text) in zip(test[0], test[1], test_text):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = crop_img(img, boxes)
    predicted_text = pipeline.process_images([img], "ENGLISH")
    wer_history.append(wer(text, predicted_text))
    cer_history.append(cer(text, predicted_text))
    print("-" * 100)
    print("Index:", i)
    print("Target:", text)
    print("Predicted:", predicted_text)
    print("Wer score:", wer_history[-1])
    print("Cer score:", cer_history[-1])
    print()
    avg_wer = sum(wer_history) / len(wer_history)
    avg_cer = sum(cer_history) / len(cer_history)
    print("Wer avg score:", avg_wer)
    print("Cer avg score:", avg_cer)
    print("-" * 100)
    i += 1
    if (i > 150):
        break