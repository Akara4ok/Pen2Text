""" Different utils function """
import sys
import numpy as np
import tensorflow as tf

def read_charlist(file_path):
    """ Read possible char lists from file """
    char_list_file = open(file_path)
    line = char_list_file.readline()
    charlist = [x for x in line]
    return charlist


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
