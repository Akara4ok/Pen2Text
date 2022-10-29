""" Pipeline for training model """
import sys
import time
from path import Path
import cv2

sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/Dataset/IAM Dataset/')
from iam_sequence import IAMSequence
sys.path.append('Pipeline/')
import model_settings as settings

sys.path.append('Pipeline/Model')
from pen2text import Pen2Text

# data_loader = DataLoaderIAM(Path("Data/IAM Dataset"), 10,
#                             settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT)

# train, val, test = data_loader.split()
# dataset = IAMSequence(train[0], train[1], settings.BATCH_SIZE, 'train')

# for (x, y) in dataset:
#     for img in x:
#         cv2.imshow("img", img)
#         cv2.waitKey(0)

char_list_file = open(settings.CHAR_DIR)
line = char_list_file.readline()
charlist = [x for x in line]

model = Pen2Text(charlist)
print("-----------------------------")
model.build(input_shape=[1, 32, 256, 1])
model.summary()