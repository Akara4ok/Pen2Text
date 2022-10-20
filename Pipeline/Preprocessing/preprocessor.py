""" Preprocessing for images """

import random
from typing import Tuple

import sys
import cv2
import numpy as np
import data_generator_settings as generator_settings

sys.path.append('Pipeline/Dataloaders/')
from dataloader_iam import Batch


class Preprocessor:
    """ Class for preprocessing for images """

    def __init__(self,
                 img_size: Tuple[int, int],
                 clustering_percent: float = 0.1,
                 data_augmentation: bool = False,
                 line_mode: bool = False) -> None:

        self.img_size = img_size
        self.data_augmentation = data_augmentation
        self.line_mode = line_mode
        self.clustering_percent = clustering_percent
        self.clustered_in_batch = 0

    @staticmethod
    def _truncate_label(text: str, max_text_len: int) -> str:
        """
        Function ctc_loss can't compute loss
        if it cannot find a mapping between text label and input
        labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        If a too-long label is provided, ctc_loss returns an infinite gradient.
        """
        cost = 0
        for (i, curr_text) in enumerate(text):
            if i != 0 and curr_text == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_len:
                return text[:i]
        return text

    def _create_text_line(self, batch: Batch) -> Batch:
        """Create image of a text line by pasting multiple word images into an image."""

        # go over all batch elements
        res_imgs = []
        res_texts = []
        for i in range(batch.batch_size):
            # number of words to put into current line
            num_words = random.randint(
                1, 8) if self.data_augmentation else generator_settings.DEFAULT_WORD_NUM

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
                    20, 50) if self.data_augmentation else generator_settings.DEFAULT_WORD_SEP
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

    def process_img(self, img: np.ndarray, batch_len: int) -> np.ndarray:
        """Clusterring, resizing and apllying data augmentation."""
        if (img is None) or (img.shape[0] <= 1 or img.shape[1] <= 1):
            img = np.zeros(self.img_size)

        res_image = img
        if self.clustered_in_batch / batch_len < self.clustering_percent:
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
        
        # general preprocessing
        scale_percent = self.img_size[0] / img.shape[0]
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        resized = cv2.resize(res_image.astype('float32'), (width, height), interpolation = cv2.INTER_AREA)

        # add padding or crop image
        if(not self.line_mode):
            color = 255
            new_height = self.img_size[0]
            new_width = self.img_size[1]
            old_height = resized.shape[0]
            old_width = resized.shape[1]
            pad_img = np.full((self.img_size[0],self.img_size[1]), color, dtype=np.uint8)

            # compute center offset
            y_center = (new_height - old_height) // 2
            x_center = max((new_width - old_width) // 2, 0)

            # copy img image into center of result image
            print(pad_img[y_center:y_center+old_height,
                x_center:x_center+old_width].shape)
            pad_img[y_center:y_center+old_height,
                x_center:x_center+old_width] = resized[:, :old_width]



        res_image = pad_img / 255
        return res_image

    def process_batch(self, batch: Batch) -> Batch:
        """ Process batch of input"""
        self.clustered_in_batch = 0
        if self.line_mode:
            batch = self._create_text_line(batch)

        res_imgs = [self.process_img(img, len(batch.imgs))
                    for img in batch.imgs]
        # res_imgs = batch[0]
        max_text_len = res_imgs[0].shape[0] // 4
        res_texts = [self._truncate_label(
            text, max_text_len) for text in batch.texts]
        return Batch(res_imgs, res_texts, batch.batch_size)
