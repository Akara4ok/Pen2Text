""" Pipeline for training model """
import sys
import cv2
import time
from path import Path

sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/Dataset/IAM Dataset/')
from iam_sequence import IAMSequence
sys.path.append('Pipeline/')
import model_settings as settings


data_loader = DataLoaderIAM(Path("Data/IAM Dataset"), 10, settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT)

train, val, test = data_loader.split()
dataset = IAMSequence(train[0], train[1], settings.BATCH_SIZE, 'test')

for (x, y) in dataset:
    for img in x:
        cv2.imshow("img", img)
        cv2.waitKey(0)