"""Class for manipulating data during training"""

from typing import Tuple
import random
import sys
import cv2
import numpy as np

from path import Path
from dataloader import DataLoader
from dataloader import Sample
from dataloader import Batch

sys.path.append('Pipeline/')
import model_settings as settings


class DataLoaderIAM(DataLoader):
    """Abstract class for manipulating data during training and infering"""

    def __init__(self, data_dir: Path, batch_size: int) -> None:
        super().__init__(data_dir, batch_size)
        self.mode = settings.TRAINING_MODE

        with open(data_dir / "ascii" / self.mode + ".txt", encoding='utf-8') as file:
            bad_samples = ['a01-117-05-02', 'r06-022-03-05']
            for line in file:
                if not line or line[0] == '#':
                    continue

                line_split = line.split(' ')
                if len(line_split) < 9:
                    raise RuntimeError("Wrong line format")

                file_name_split = line_split[0].split('-')
                file_name_subdir1 = file_name_split[0]
                file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
                file_base_name = line_split[0] + '.png'
                file_name = data_dir / 'images' / self.mode / \
                    file_name_subdir1 / file_name_subdir2 / file_base_name

                if line_split[0] in bad_samples:
                    print('Ignoring known broken image:', file_name)
                    continue

                text = ' '.join(line_split[8:])
                self.samples.append(Sample(text, file_name))

            train_idx = int(settings.TRAIN_PERCENT * len(self.samples))
            val_idx = train_idx + \
                int(settings.VAL_PERCENT * len(self.samples))

            self.train_samples = self.samples[:train_idx]
            random.shuffle(self.train_samples)
            self.val_samples = self.samples[train_idx:val_idx]
            self.test_samples = self.samples[val_idx:]

            self.train_curr_idx = 0
            self.val_curr_idx = 0
            self.test_curr_idx = 0

            self.train_words = [x.text for x in self.train_samples]
            self.val_words = [x.text for x in self.val_samples]
            self.testing_words = [x.text for x in self.test_samples]

    def get_iterator_info(self) -> Tuple[int, int]:
        num_batches = int(np.ceil(len(self.samples) / self.batch_size))
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        # val set: allow last batch to be smaller
        return self.curr_idx < len(self.samples)

    def has_train_batch(self) -> bool:
        """Is there a next element for training?"""
        return self.train_curr_idx < len(self.train_samples)

    def has_val_batch(self) -> bool:
        """Is there a next element for validation?"""
        return self.val_curr_idx < len(self.val_samples)

    def has_test_batch(self) -> bool:
        """Is there a next element for testing?"""
        return self.test_curr_idx < len(self.test_samples)

    def _get_img(self, i: int, mode: str = '') -> np.ndarray:
        if mode == '':
            img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)
        if mode == 'train':
            img = cv2.imread(self.train_samples[i].file_path, cv2.IMREAD_GRAYSCALE)
        if mode == 'val':
            img = cv2.imread(self.val_samples[i].file_path, cv2.IMREAD_GRAYSCALE)
        if mode == 'test':
            img = cv2.imread(self.test_samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_next(self) -> Batch:
        batch_range = range(self.curr_idx, min(
            self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        texts = [self.samples[i].text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, texts, len(imgs))

    def get_train_batch(self) -> Batch:
        """ Returns train batch """
        batch_range = range(self.train_curr_idx, min(
            self.train_curr_idx + self.batch_size, len(self.train_samples)))

        imgs = [self._get_img(i, 'train') for i in batch_range]
        texts = [self.train_samples[i].text for i in batch_range]

        self.train_curr_idx += self.batch_size
        return Batch(imgs, texts, len(imgs))

    def get_val_batch(self) -> Batch:
        """ Returns train batch """
        batch_range = range(self.val_curr_idx, min(
            self.val_curr_idx + self.batch_size, len(self.val_samples)))

        imgs = [self._get_img(i, 'val') for i in batch_range]
        texts = [self.val_samples[i].text for i in batch_range]

        self.val_curr_idx += self.batch_size
        return Batch(imgs, texts, len(imgs))

    def get_test_batch(self) -> Batch:
        """ Returns train batch """
        batch_range = range(self.test_curr_idx, min(
            self.test_curr_idx + self.batch_size, len(self.test_samples)))

        imgs = [self._get_img(i, 'test') for i in batch_range]
        texts = [self.test_samples[i].text for i in batch_range]

        self.test_curr_idx += self.batch_size
        return Batch(imgs, texts, len(imgs))
