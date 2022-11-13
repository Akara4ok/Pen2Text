""" Preprocessing for images """

from email.policy import default
import random
from typing import Tuple

import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/utils')
from utils_types import Batch
from utils_types import Sample
from utils import get_img


class RecognitionPreprocessor:
    """ Class for preprocessing for images """

    def __init__(self,
                 img_size: Tuple[int, int],
                 char_list: list,
                 max_len: int,
                 batch_size: int,
                 clustering_percent: float = 1,
                 data_augmentation: bool = False,
                 line_mode: bool = False) -> None:

        self.img_size = img_size
        self.data_augmentation = data_augmentation
        self.line_mode = line_mode
        self.clustering_percent = clustering_percent
        self.clustered_in_batch = 0
        self.char_list = char_list
        self.batch_size = batch_size
        self.max_len = max_len

    def _create_text_line(self, batch: Batch) -> Batch:
        """Create image of a text line by pasting multiple word images into an image."""

        # go over all batch elements
        res_imgs = []
        res_texts = []
        for i in range(batch.batch_size):
            # number of words to put into current line
            num_words = random.randint(
                1, 8) if self.data_augmentation else settings.DEFAULT_WORD_NUM

            # concat ground truth texts
            curr_res_text = ' '.join(
                [batch.texts[(i + j) % batch.batch_size] for j in range(num_words)])
            res_texts.append(curr_res_text)

            # put selected word images into list, compute target image size
            sel_imgs = []
            word_seps = [0]
            height = 0
            width = 0
            for j in range(num_words):
                curr_sel_img = batch.imgs[(i + j) % batch.batch_size]
                curr_word_sep = random.randint(
                    20, 50) if self.data_augmentation else settings.DEFAULT_WORD_SEP
                height = max(height, curr_sel_img.shape[0])
                width += curr_sel_img.shape[1]
                sel_imgs.append(curr_sel_img)
                if j + 1 < num_words:
                    width += curr_word_sep
                    word_seps.append(curr_word_sep)

            # put all selected word images into target image
            target = np.ones([height, width], np.uint8) * 255
            curr_width = 0
            for curr_sel_img, curr_word_sep in zip(sel_imgs, word_seps):
                curr_width += curr_word_sep
                curr_height = (height - curr_sel_img.shape[0]) // 2
                target[curr_height:curr_height + curr_sel_img.shape[0]:, curr_width:curr_width +
                       curr_sel_img.shape[1]] = curr_sel_img
                curr_width += curr_sel_img.shape[1]

            # put image of line into result
            res_imgs.append(target)

        return Batch(res_imgs, res_texts, batch.batch_size)

    def process_img(self, img: np.ndarray) -> np.ndarray:
        """Clusterring, resizing and apllying data augmentation."""
        if (img is None) or (img.shape[0] <= 1 or img.shape[1] <= 1) or (img.shape[0] / img.shape[1] < 0.05):
            img = np.zeros(self.img_size)

        res_image = img
        if self.clustered_in_batch / self.batch_size < self.clustering_percent:
            # numpy reshape operation -1 unspecified
            pixel_vals = img.reshape((-1, 1))
            pixel_vals = np.float32(pixel_vals)
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            k = 2

            _, labels, centers = cv2.kmeans(
                pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            if centers[0] > centers[1]:
                labels = 1 - labels

            centers = np.array([[0], [255]])

            # Mapping labels to center points( RGB Value)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((img.shape))
            self.clustered_in_batch += 1
            res_image = segmented_image

        if self.data_augmentation:
            height, width = self.img_size
            current_height, current_width = res_image.shape
            # photometric data augmentation
            # TODO delete this part if accuracy will be low
            if random.random() < 0.25:
                def gaussian_koef():
                    return random.randint(1, 3) * 2 + 1
                img = cv2.GaussianBlur(
                    img, (gaussian_koef(), gaussian_koef()), 0)
            if random.random() < 0.25:
                img = cv2.dilate(img, np.ones((3, 3)))

            # geometric data augmentation
            resized_koef = min(width / current_width, height / current_height)
            resized_koef_x = resized_koef * np.random.uniform(0.75, 1.05)
            resized_koef_y = resized_koef * np.random.uniform(0.75, 1.05)

            # random position around center
            low_xc = (width - current_width * resized_koef_x) / 2
            low_yc = (height - current_height * resized_koef_y) / 2
            clipped_xc = max((width - current_width * resized_koef_x) / 2, 0)
            clipped_yc = max((height - current_height * resized_koef_y) / 2, 0)
            xc_bias = low_xc + np.random.uniform(-clipped_xc, clipped_xc)
            yc_bias = low_yc + np.random.uniform(-clipped_yc, clipped_yc)
            transform_matrix = np.float32(
                [[resized_koef_x, 0, xc_bias], [0, resized_koef_y, yc_bias]])
            target = np.full(
                (self.img_size[0], self.img_size[1]), 255, dtype=np.uint8)

            res_image = cv2.warpAffine(res_image.astype(np.uint8), transform_matrix,
                                       dsize=(
                                           self.img_size[1], self.img_size[0]),
                                       flags=cv2.INTER_AREA, dst=target,
                                       borderMode=cv2.BORDER_TRANSPARENT)

        else:
            # general preprocessing
            old_width = img.shape[1]
            old_height = img.shape[0]
            scale_percent = min(
                self.img_size[1] / old_width, self.img_size[0] / old_height)

            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            resized = cv2.resize(res_image.astype('float32'),
                                 (width, height), interpolation=cv2.INTER_AREA)
            # add padding or crop image
            if not self.line_mode:
                color = 255
                new_height = self.img_size[0]
                new_width = self.img_size[1]
                old_height = resized.shape[0]
                old_width = resized.shape[1]
                pad_img = np.full(
                    (self.img_size[0], self.img_size[1]), color, dtype=np.uint8)

                # compute center offset
                y_center = max((new_height - old_height) // 2, 0)
                x_center = max((new_width - old_width) // 2, 0)

                pad_img[y_center:y_center+old_height,
                        x_center:x_center+old_width] = resized
                res_image = pad_img

        res_image = res_image / 255
        res_image = np.expand_dims(res_image,axis=-1)
        return res_image

    def process_text(self, text):
        """ Process text """
        #print("---------", text, "-----------")
        processed_text = []
        for char in text:
            try:
                processed_text.append(self.char_list.index(char))
            except:
                pass
        #print("---------", processed_text, "-----------")
        return processed_text

    def process_single(self, path: tf.Tensor, text: tf.Tensor) -> Sample:
        """ Process single img and text to img """
        img = get_img(path.numpy().decode("utf-8"))
        res_img = self.process_img(img)
        res_text = self.process_text(text.numpy().decode("utf-8"))
        return (res_img, res_text)

    def process_batch(self, batch: Batch) -> Batch:
        """ Process batch of input"""
        self.clustered_in_batch = 0
        if self.line_mode:
            batch = self._create_text_line(batch)

        res_imgs = np.array([self.process_img(img)
                             for img in batch.imgs])

        res_texts = np.array([self.process_text(text)
                              for text in batch.texts])
        res_texts = tf.keras.utils.pad_sequences(
            res_texts, maxlen=self.max_len, padding='post', value=len(self.char_list))

        return Batch(res_imgs, res_texts)
