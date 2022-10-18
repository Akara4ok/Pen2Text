""" Pipeline for training model """
import sys
import time
import cv2
from path import Path
sys.path.append('Pipeline/Dataloaders/')
from dataloader_iam import DataLoaderIAM

sys.path.append('Pipeline/Preprocessing/')
from preprocessor import Preprocessor

data_loader = DataLoaderIAM(Path("Data/IAM Dataset"), 100)

preprocessor = Preprocessor((256, 32), True, False)

start = time.time()
i = 0
while(data_loader.has_next):
    old_images = data_loader.get_train_batch()

    new_images = preprocessor.process_batch(old_images)
    print("batch -",  i)
    i += 1
    if(i == 100):
        break

end = time.time()
print(end - start)
