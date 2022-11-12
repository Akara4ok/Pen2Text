""" Inference """

import sys
from path import Path
import tensorflow as tf
import numpy as np

sys.path.append('Pipeline/Dataloaders/InferenceDataloader/')
from inference_dataloader import InferenceDataloader
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/utils')
from utils import simple_decode
from utils import read_charlist
sys.path.append('Pipeline/Preprocessing')
from preprocessor import Preprocessor

char_list = read_charlist("./Pipeline/CharList.txt")
max_len = settings.MAX_LEN

model_path = "./Models/Test/tf"
model = tf.keras.models.load_model(model_path, compile=False)

dataloader = InferenceDataloader(Path("TestInference/Recognitions"))
filenames = dataloader.get_filenames()
imgs = dataloader.get_images()

preprocessor = Preprocessor(img_size=(settings.HEIGHT, settings.WIDTH), char_list=char_list, max_len=max_len, batch_size=settings.BATCH_SIZE)
processed_imgs = [preprocessor.process_img(img) for img in imgs]

labels = []

predictions = model.predict(processed_imgs, verbose=0)
out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                        greedy=False)[0][0])
for x in out:
    batch_correct = 0
    label = simple_decode(x, char_list)
    labels.append(label)
    
for i, label in enumerate(labels):
    print("-" * 100)
    print(filenames[i])
    print("predicted text =", label)