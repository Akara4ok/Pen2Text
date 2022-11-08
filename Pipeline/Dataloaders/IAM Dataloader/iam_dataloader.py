"""Class for loading and spliting IAM dataset"""

import sys
from typing import Tuple
from path import Path
sys.path.append('Pipeline/')
import model_settings as settings
sys.path.append('Pipeline/Dataloaders/')
from dataloader import DataLoader
sys.path.append('Pipeline/utils')
from utils_types import Sample


class DataLoaderIAM(DataLoader):
    """Class for loading and spliting IAM dataset"""

    def __init__(self, data_dir: Path, batch_size: int, train_percent: float,
                    val_percent: float, test_percent: float, img_num: int) -> None:
        super().__init__(data_dir, batch_size, train_percent, val_percent, test_percent, img_num)
        self.mode = 'lines' if settings.LINE_MODE else 'words'

        with open(data_dir / "ascii" / self.mode + ".txt", encoding='utf-8') as file:
            bad_samples = ['a01-117-05-02', 'r06-022-03-05']
            i = 0
            for line in file:
                if (img_num != -1 and i > img_num):
                    break
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
                i += 1

    def split(self) -> Tuple:
        train_idx = int(settings.TRAIN_PERCENT * len(self.samples))
        val_idx = train_idx + \
            int(settings.VAL_PERCENT * len(self.samples))

        train_samples = self.samples[:train_idx]
        val_samples = self.samples[train_idx:val_idx]
        test_samples = self.samples[val_idx:]

        train_paths = [x.file_path for x in train_samples]
        val_paths = [x.file_path for x in val_samples]
        test_paths = [x.file_path for x in test_samples]

        train_words = [x.text for x in train_samples]
        val_words = [x.text for x in val_samples]
        test_words = [x.text for x in test_samples]
        return ((train_paths, train_words),
                (val_paths, val_words),
                (test_paths, test_words))
