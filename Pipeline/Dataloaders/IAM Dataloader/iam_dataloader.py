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
from dataset import Form
from dataset import Line
from dataset import Word
import random

class DataLoaderIAM(DataLoader):
    """Class for loading and spliting IAM dataset"""

    def __init__(self, data_dir: Path, train_percent: float,
                    val_percent: float, test_percent: float, img_num: int) -> None:
        super().__init__(data_dir, train_percent, val_percent, test_percent, img_num)
        self.mode = 'lines' if settings.LINE_MODE else 'words'
        self.max_len = 0

        self.dataset = {}


        with open(data_dir / "ascii" / "forms" + ".txt", encoding='utf-8') as file:
            for line in file:
                bad_samples = []
                if not line or line[0] == '#':
                    continue

                line_split = line.split(' ')
                if len(line_split) < 8:
                    raise RuntimeError("Wrong line format")

                if line_split[0] in bad_samples:
                    print('Ignoring known broken image:', file_name)
                    continue

                form = Form(data_dir, line_split)
                self.dataset[form.id] = form

        
        with open(data_dir / "ascii" / "lines" + ".txt", encoding='utf-8') as file:
            bad_samples = []
            for line in file:
                if not line or line[0] == '#':
                    continue

                line_split = line.split(' ')
                if len(line_split) < 9:
                    raise RuntimeError("Wrong line format")

                if line_split[0] in bad_samples:
                    print('Ignoring known broken image:', line_split[0])
                    continue

                line = Line(data_dir, line_split)
                self.dataset[line.form_id].add_line(line.id, line)


        self.word_form_id = []
        self.word_line_id = []
        self.word_bounding_boxes = []

        with open(data_dir / "ascii" / "words" + ".txt", encoding='utf-8') as file:
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

                if(len(text) == 54):
                    continue

                if line_split[3] == '-1':
                    continue

                word = Word(data_dir, line_split)
                self.dataset[word.form_id].add_word(word.line_id, word.id, word)

                if (self.max_len < len(text)):
                    self.max_len = len(text) 
                self.samples.append(Sample(text, file_name))
                i += 1

    def get_max_len(self) -> int:
        """ Return max text len """
        return self.max_len

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

        train_paths = [x.file_path for x in train_samples]
        val_paths = [x.file_path for x in val_samples]
        test_paths = [x.file_path for x in test_samples]

        train_words = [x.text for x in train_samples]
        val_words = [x.text for x in val_samples]
        test_words = [x.text for x in test_samples]
        return ((train_paths, train_words),
                (val_paths, val_words),
                (test_paths, test_words))

    def split_for_page_segmentation(self, shuffle = True, random_seed = None) -> Tuple:
        """ Split dataset to train, validation and test """
        forms = list(self.dataset.values())
        if shuffle:
            if random_seed:
                random.Random(random_seed).shuffle(forms)
            else:
                random.shuffle(forms)

        
        train_idx = int(settings.TRAIN_PERCENT * len(forms))
        val_idx = train_idx + \
            int(settings.VAL_PERCENT * len(forms))
        
        train_forms_path = [x.file_name for x in forms[:train_idx]]
        train_line_boxes = [x.get_line_boxes() for x in forms[:train_idx]]

        val_forms_path = [x.file_name for x in forms[train_idx:val_idx]]
        val_line_boxes = [x.get_line_boxes() for x in forms[train_idx:val_idx]]

        test_forms_path = [x.file_name for x in forms[val_idx:]]
        test_line_boxes = [x.get_line_boxes() for x in forms[val_idx:]]

        return ((train_forms_path, train_line_boxes),
                (val_forms_path, val_line_boxes),
                (test_forms_path, test_line_boxes))
    
        
    def split_for_line_segmentation(self, shuffle = True, random_seed = None) -> Tuple:
        forms = list(self.dataset.values())
        lines = []
        for form in forms:
            lines.extend(form.lines.values())

        if shuffle:
            if random_seed:
                random.Random(random_seed).shuffle(lines)
            else:
                random.shuffle(lines)

        
        train_idx = int(settings.TRAIN_PERCENT * len(lines))
        val_idx = train_idx + \
            int(settings.VAL_PERCENT * len(lines))
        
        train_forms_path = [self.dataset[x.form_id].file_name for x in lines[:train_idx]]
        train_line_box = [x.get_box() for x in lines[:train_idx]]
        train_word_boxes = [x.get_word_boxes() for x in lines[:train_idx]]

        val_forms_path = [self.dataset[x.form_id].file_name for x in lines[train_idx:val_idx]]
        val_line_box = [x.get_box() for x in lines[train_idx:val_idx]]
        val_word_boxes = [x.get_word_boxes() for x in lines[train_idx:val_idx]]

        test_forms_path = [self.dataset[x.form_id].file_name for x in lines[val_idx:]]
        test_line_box = [x.get_box() for x in lines[val_idx:]]
        test_word_boxes = [x.get_word_boxes() for x in lines[val_idx:]]

        return ((train_forms_path, train_line_box, train_word_boxes),
                (val_forms_path, val_line_box, val_word_boxes),
                (test_forms_path, test_line_box, test_word_boxes))
        

    def split_for_word_segmentation(self, shuffle = True, random_seed = None) -> Tuple:
        """ Split dataset to train, validation and test """
        forms = list(self.dataset.values())
        if shuffle:
            if random_seed:
                random.Random(random_seed).shuffle(forms)
            else:
                random.shuffle(forms)

        
        train_idx = int(settings.TRAIN_PERCENT * len(forms))
        val_idx = train_idx + \
            int(settings.VAL_PERCENT * len(forms))
        
        train_forms_path = [x.file_name for x in forms[:train_idx]]
        train_line_boxes = [x.get_word_boxes() for x in forms[:train_idx]]

        val_forms_path = [x.file_name for x in forms[train_idx:val_idx]]
        val_line_boxes = [x.get_word_boxes() for x in forms[train_idx:val_idx]]

        test_forms_path = [x.file_name for x in forms[val_idx:]]
        test_line_boxes = [x.get_word_boxes() for x in forms[val_idx:]]

        return ((train_forms_path, train_line_boxes),
                (val_forms_path, val_line_boxes),
                (test_forms_path, test_line_boxes))