""" Different utils function """

import numpy as np
import tensorflow as tf
import os
import cv2

def read_charlist(file_path: str):
    """ Read possible char lists from file """
    char_list_file = open(file_path)
    line = char_list_file.readline()
    charlist = [x for x in line]
    return charlist

def simple_decode(pred: tf.Tensor, char_list: list) -> str:
    """ Decode one sample with simple decoding """
    res = ''
    for c in pred:
        if int(c) != -1 and int(c) != len(char_list):
            res += char_list[int(c)]
    return res


def decode_batch_predictions(pred: tf.Tensor, num_to_char: tf.keras.layers) -> list:
    """ Decode ctc values to text """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

def last_checkpoint(checkpoint_dir, begin_pos = 3, end_pos = 7, filename_end_pos = 12):
    """ Get last checkpoint in directory """
    res_filename = ""
    max_epoch = -1
    for filename in os.listdir(checkpoint_dir):
        try:
            epoch_number = filename[begin_pos:end_pos]
            if(max_epoch < int(epoch_number)):
                max_epoch = int(epoch_number)
                res_filename = filename[:filename_end_pos]
        except:
            pass
    return max_epoch, res_filename


def get_img(path):
    """ Read image """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img