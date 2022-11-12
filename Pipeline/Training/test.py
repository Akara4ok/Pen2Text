""" Pipeline for training model """
import sys
from path import Path
import tensorflow as tf
import numpy as np

sys.path.append('Pipeline/Dataloaders/IAM Dataloader/')
from iam_dataloader import DataLoaderIAM
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/Model')
from pen2text import Pen2Text
from improved_ocr import ImprovedPen2Text
sys.path.append('Pipeline/utils')
from utils import read_charlist
sys.path.append('Pipeline/Preprocessing')
from preprocessor import Preprocessor
sys.path.append('Pipeline/utils')
from utils import simple_decode

#read dataset
data_loader = DataLoaderIAM(Path("Data/IAM Dataset"), 10,
                            settings.TRAIN_PERCENT, settings.VAL_PERCENT, settings.TEST_PERCENT, settings.IMG_NUM)
train, val, test = data_loader.split()

char_list = read_charlist("./Pipeline/CharList.txt")
max_len = data_loader.get_max_len()

preprocessor = Preprocessor(img_size=(settings.HEIGHT, settings.WIDTH), char_list=char_list, max_len=max_len, batch_size=settings.BATCH_SIZE)

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

model=ImprovedPen2Text(char_list)
model.build((None,32,128,1))
model.load_weights("./Checkpoints/Test/cp-0001.ckpt")

correct = 0
for batch_index, batch in enumerate(test_dataset):
    X, y = batch
    predictions = model.predict(X)
    targets = []
    targets.extend(y.numpy())
    out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                            greedy=False)[0][0])
    for i, x in enumerate(out):
        target = simple_decode(y[i], char_list)
        label = simple_decode(x, char_list)
        print("Batch index:", batch_index)
        if(label == target):
            correct += 1
            print("-----Correct-----")
        else:
            print("-----Not correct-----")
        print("original_text =", target)
        print("predicted text =", label)
        print("-" * 100)

wer_score = 1 - correct / len(test_dataset)
print("-" * 100)
print(f"Word Error Rate: {wer_score:.4f}")
print("-" * 100)