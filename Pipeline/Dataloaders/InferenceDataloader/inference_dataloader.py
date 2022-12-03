"""Class for loading and spliting IAM dataset"""

import os
import sys
from typing import Tuple
from path import Path
sys.path.append('Pipeline/Dataloaders/')
from dataloader import DataLoader
sys.path.append('Pipeline/utils/')
from utils import get_img


class InferenceDataloader(DataLoader):
    """Class for loading and spliting IAM dataset"""

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir, 0, 0, 1, None)
        self.filenames = []
        for filename in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, filename)
            self.filenames.append(filename)
            self.samples.append(get_img(full_path))

    def get_filenames(self) -> list:
        """ Return images from dataloader """
        return self.filenames

    def get_images(self) -> list:
        """ Return images from dataloader """
        return self.samples
