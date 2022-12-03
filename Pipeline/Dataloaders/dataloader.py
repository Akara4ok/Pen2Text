"""Abstract class for loading and splitting data"""

import abc
import errno
import os

from typing import Tuple
from path import Path


class DataLoader(metaclass=abc.ABCMeta):
    """Abstract class for loading and splitting data"""

    def __init__(self, data_dir: Path, train_percent: float, val_percent: float, test_percent: float, img_num: int) -> None:
        if not data_dir.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), data_dir)

        if abs(train_percent + val_percent + test_percent - 1) > 0.001:
            raise ValueError("Sum of percent is not equal to 1")

        self.data_dir = data_dir
        self.curr_idx = 0
        self.img_num = img_num

        self.samples = []