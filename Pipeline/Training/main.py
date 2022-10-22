""" Pipeline for training model """
import sys
import cv2
import time
from path import Path

sys.path.append('Pipeline/Dataloaders/')
from dataloader_iam import DataLoaderIAM
sys.path.append('Pipeline/Preprocessing/')
from preprocessor import Preprocessor
sys.path.append('Pipeline/')
import model_settings as settings

data_loader = DataLoaderIAM(Path("Data/IAM Dataset"), 100)

preprocessor = Preprocessor((settings.WIDTH, settings.HEIGHT), False, False)

start = time.time()
i = 0
while data_loader.has_test_batch():
    old_images = data_loader.get_test_batch()

    new_images = preprocessor.process_batch(old_images)
    print("batch -",  i)
    i += 1

end = time.time()
print(end - start)
