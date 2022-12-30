""" Pipeline for testing model """
import sys
from path import Path
import tensorflow as tf
import numpy as np
import time
from jiwer import cer

sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/utils')
from utils import read_charlist
sys.path.append('Pipeline/Preprocessing')
from recognition_preprocessor import RecognitionPreprocessor
sys.path.append('Pipeline/utils')
from utils import simple_decode
sys.path.append('Pipeline/Postprocessing')
from spell_correction.spell_correction import correction

#read dataset
data_loader = DataLoaderIAM(Path("Data/IAM Dataset"),
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)
train, val, test = data_loader.split_for_recognition(shuffle=False)

char_list = read_charlist("./Pipeline/CharList.txt")
max_len = data_loader.get_max_len()

preprocessor = RecognitionPreprocessor(img_size=(settings.HEIGHT, settings.WIDTH), char_list=char_list, max_len=max_len, batch_size=settings.BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (val[0], val[1])
    ).map(
        lambda x, y: tf.py_function(preprocessor.process_single, [x, y], [tf.float32, tf.uint8]), 
        num_parallel_calls=tf.data.AUTOTUNE
        ).padded_batch(
            settings.BATCH_SIZE, 
            padded_shapes=([None, None, 1], [None]),
            padding_values=(0., tf.cast(len(char_list), dtype=tf.uint8))
            ).prefetch(buffer_size=tf.data.AUTOTUNE)

model_name = "ImprovedPen2Text_v7"
model=tf.keras.models.load_model("./Models/Recognition/Models/" + model_name + "/tf", compile=False)

correct = 0
targets = []
labels = []
begin = time.time()
for batch_index, batch in enumerate(test_dataset):
    X, y = batch
    predictions = model.predict(X, verbose=0)
    out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                            greedy=False)[0][0])
    for i, x in enumerate(out):
        batch_correct = 0
        target = simple_decode(y[i], char_list)
        label = simple_decode(x, char_list)
        # print("-" * 100)
        # print("Original:", target)
        # print("Predicted:", label)
        if(label.isalpha()):
            label = correction(label)
            # print("Corrected:", label)
        targets.append(target)
        labels.append(label)
        if(label == target):
            correct += 1

end = time.time()

wer_score = 1 - correct / (len(test_dataset) * settings.BATCH_SIZE)
print("-" * 100)
print(f"Word Error Rate: {wer_score:.4f}")

cer_score = cer(targets, labels)
print("-" * 100)
print(f"Character Error Rate: {cer_score:.4f}")

total_time = end - begin
print("-" * 100)
print(f"Total time: {total_time:.4f}")
print("-" * 100)