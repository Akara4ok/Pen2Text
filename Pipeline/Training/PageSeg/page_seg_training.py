""" Pipeline for training model for page segmentation to line """
import sys
from path import Path
import tensorflow as tf
import cv2
import numpy as np
sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/utils')
from utils import read_charlist
from utils import get_img
sys.path.append('Pipeline/Preprocessing')
from page_seg_preprocessing import PageSegPreprocessor

data_loader = DataLoaderIAM(Path("Data/IAM Dataset"),
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)

train, val, test = data_loader.split_for_line_segmentation(random_seed=settings.RANDOM_SEED)
preprocessor = PageSegPreprocessor(batch_size=settings.BATCH_SIZE, img_size=(512, 512))

i = 0
for sample in zip(train[0], train[1], train[2]):
    print(sample)
    img = get_img(sample[0])
    img = cv2.resize(img, (512, 512))
    cv2.imshow("img", img)
    cv2.waitKey(0)
    i += 1
    if i == 5:
        break