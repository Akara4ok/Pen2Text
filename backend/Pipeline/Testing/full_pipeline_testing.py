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
predicted_texts = []
target_texts = []
with open("Logs/model_testing.txt", "w") as text_file:
    for (path, boxes, text) in zip(test[0], test[1], test_text):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = crop_img(img, boxes)
        predicted_texts += pipeline.process_images([img], "ENGLISH_LETTERS_NUMBERS")
        target_texts.append(text)
        print("-" * 100)
        print("Index:", i)
        print("Target:", text)
        print("Predicted:", predicted_texts[-1])
        print("Wer score:", wer(text, predicted_texts[-1]))
        print("Cer score:", cer(text, predicted_texts[-1]))

        print("-" * 100, file=text_file)
        print("Index:", i, file=text_file)
        print("Target:", text, file=text_file)
        print("Predicted:", predicted_texts[-1], file=text_file)
        print("Wer score:", wer(text, predicted_texts[-1]), file=text_file)
        print("Cer score:", cer(text, predicted_texts[-1]), file=text_file)
        i += 1
        if(i > 200):
            break

    avg_wer = wer(target_texts, predicted_texts)
    avg_cer = cer(target_texts, predicted_texts)

    print("Wer avg score:", avg_wer, file=text_file)
    print("Cer avg score:", avg_cer, file=text_file)