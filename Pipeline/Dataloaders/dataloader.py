"""Abstract class for manipulating data during training and infering"""

import abc
from collections import namedtuple
import errno
import os
import sys

from typing import Tuple
from path import Path
import numpy as np

sys.path.append('Pipeline/')
import model_settings as settings


Sample = namedtuple('Sample', 'text, file_path')
Batch = namedtuple('Batch', 'imgs, texts, batch_size')


class DataLoader(metaclass=abc.ABCMeta):
    """Abstract class for manipulating data during training and infering"""

    def __init__(self, data_dir: Path, batch_size: int) -> None:
        if not data_dir.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), data_dir)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.curr_idx = 0

        self.samples = []
        self.char_list = [ch for ch in open(
            settings.CHAR_DIR, encoding='UTF-8').read() if ch != '\n' if ch != ' ']

    @abc.abstractmethod
    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""

    @abc.abstractmethod
    def has_next(self) -> bool:
        """Is there a next element?"""

    @abc.abstractmethod
    def _get_img(self, i: int) -> np.ndarray:
        """Is there a next element?"""

    @abc.abstractmethod
    def get_next(self) -> Batch:
        """Get next element."""
