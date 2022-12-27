"""Class for loading and spliting ukrainian words dataset"""

import sys
from typing import Tuple
from path import Path
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/Dataloaders/')
from dataloader import DataLoader
sys.path.append('Pipeline/utils')
from utils_types import Sample
import random
import pandas as pd

class UkrDataloader(DataLoader):
    """Class for loading and spliting ukrainian words dataset"""
    def __init__(self, data_dir: Path, train_percent: float,
                    val_percent: float, test_percent: float, img_num: int) -> None:
        super().__init__(data_dir, train_percent, val_percent, test_percent, img_num)
        data = pd.read_csv(self.data_dir + "/" + "words.csv", sep=',', encoding='utf-8')
        self.samples = list(zip(data.path, data.word))
    
    def split_for_recognition(self, shuffle = True, random_seed = None) -> Tuple:
        """ Split dataset to train, validation and test """
        if shuffle:
            if random_seed:
                random.Random(random_seed).shuffle(self.samples)
            else:
                random.shuffle(self.samples)

        train_idx = int(settings.TRAIN_PERCENT * len(self.samples))
        val_idx = train_idx + \
            int(settings.VAL_PERCENT * len(self.samples))

        train_samples = self.samples[:train_idx]
        val_samples = self.samples[train_idx:val_idx]
        test_samples = self.samples[val_idx:]

        train_paths = [x[0] for x in train_samples]
        val_paths = [x[0] for x in val_samples]
        test_paths = [x[0] for x in test_samples]

        train_words = [x[1] for x in train_samples]
        val_words = [x[1] for x in val_samples]
        test_words = [x[1] for x in test_samples]
        return ((train_paths, train_words),
                (val_paths, val_words),
                (test_paths, test_words))