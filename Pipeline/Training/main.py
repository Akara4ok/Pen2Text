""" Pipeline for training model """
import sys
import cv2
from path import Path
sys.path.append('Pipeline/Dataloaders/')
from dataloader_iam import DataLoaderIAM
sys.path.append('Pipeline/Preprocessing/')
from preprocessor import Preprocessor

data_loader = DataLoaderIAM(Path("Data/IAM Dataset"), 10)

preprocessor = Preprocessor((32, 256), 0.1, False, False)

old_images = data_loader.get_train_batch()

new_images = preprocessor.process_batch(old_images)

for old_image in old_images[0]:
    cv2.imshow("old image", old_image)
    cv2.waitKey(0)

for (index, new_image) in enumerate(new_images[0]):
    print(new_images[1][index])
    cv2.imshow("new image", new_image)
    cv2.waitKey(0)
