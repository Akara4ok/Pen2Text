""" Different utils function """

import numpy as np
import tensorflow as tf
import os
import cv2
import random
import sys
sys.path.append('Pipeline/')
import model_settings as settings

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

def custom_image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """ Resizeing image without distorion """
    (h, w) = image.shape[:2]
    dim = custom_image_resize_sizes(h, w, width, height, inter)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def custom_image_resize_sizes(h, w, new_width = None, new_height = None, inter = cv2.INTER_AREA):
    """ Calculating size for image resizing """
    dim = None
    if new_width is None and new_height is None:
        return (h, w)

    if new_width is None:
        r = new_height / float(h)
        dim = (int(w * r), new_height)
    else:
        r = new_width / float(w)
        dim = (new_width, int(h * r))
    
    return dim

def pad_or_resize(image, new_width, new_height):
    (height, width) = image.shape[:2]

    if(width > new_width):
        image = custom_image_resize(image, width=new_width)
    else:
        to_pad = np.zeros((height,new_width-width))
        image = np.concatenate((image,to_pad),axis=1)
    
    (height, width) = image.shape[:2]

    if(height > new_height):
        image = custom_image_resize(image, height=new_height)
    else:
        to_pad=np.zeros((new_height-height,width))
        image=np.concatenate((image,to_pad), axis=0)
    
    return image

def crop_img(img, bounding_boxes):
    """ Create mask from bounding boxes """
    min_x = 100000
    min_y = 100000
    max_x = -1
    max_y = -1
    
    for x, y, w, h in bounding_boxes:
        if max_x < x + w:
            max_x = x + w
        if min_x > x:
            min_x = x
        if max_y < y + h:
            max_y = y + h
        if min_y > y:
            min_y = y

    img = img[min_y:max_y, min_x:max_x]
    return img