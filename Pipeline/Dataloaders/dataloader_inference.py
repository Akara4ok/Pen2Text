"""Class for manipulating data during infering"""

from typing import Tuple
from path import Path

import cv2
import numpy as np

from dataloader import DataLoader
from dataloader import Batch


class DataLoaderInference(DataLoader):
    """Abstract class for manipulating data during infering"""

    def __init__(self, data_dir: Path, batch_size: int) -> None:
        super().__init__(data_dir, batch_size)

    def get_iterator_info(self) -> Tuple[int, int]:
        num_batches = int(np.ceil(len(self.samples) / self.batch_size))
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        # val set: allow last batch to be smaller
        return self.curr_idx < len(self.samples)

    def _get_img(self, i: int) -> np.ndarray:
        img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)

        return img

    def get_next(self) -> Batch:
        batch_range = range(self.curr_idx, min(
            self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        texts = [self.samples[i].text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, texts, len(imgs))
