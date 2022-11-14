"""Classes for representing iam dataset"""

import os

class Form():
    """Class for representing forms"""

    def __init__(self, data_dir: str, line_split: str) -> None:
        self.id = line_split[0]
        self.file_name = data_dir / "images" / "forms" / line_split[0] + ".png"
        self.lines = {}
    
    def add_line(self, id, line):
        self.lines[id] = line

    def add_word(self, line_id, word_id, word):
        self.lines[line_id].add_word(word_id, word)
    
    def get_line_boxes(self):
        line_array = list(self.lines.values())
        bounding_boxes = []
        for line in line_array:
            bounding_boxes.append((line.x, line.y, line.width, line.height))
        return bounding_boxes

class Line():
    """Class for representing lines"""

    def __init__(self, data_dir: str, line_split: str) -> None:
        file_name_split = line_split[0].split('-')
        file_name_subdir1 = file_name_split[0]
        file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
        file_base_name = line_split[0] + '.png'
        file_name = data_dir / 'images' / "lines" / \
            file_name_subdir1 / file_name_subdir2 / file_base_name + ".png"

        self.id = file_name_split[2]
        self.form_id = file_name_split[0] + "-" + file_name_split[1]
        self.full_id = line_split[0]
        self.x = int(line_split[4])
        self.y = int(line_split[5])
        self.width = int(line_split[6])
        self.height = int(line_split[7])
        self.words = {}
        self.file_name = file_name

    def add_word(self, id, word):
        self.words[id] = word

    def get_box(self):
        return (self.x, self.y, self.width, self.height)

    def get_word_boxes(self):
        word_array = list(self.words.values())
        bounding_boxes = []
        for word in word_array:
            bounding_boxes.append((word.x, word.y, word.width, word.height))
        return bounding_boxes

class Word():
    """Class for representing words"""

    def __init__(self, data_dir: str, line_split: str) -> None:
        self.id = line_split[0]
        self.filename = data_dir / "images" / "forms" / line_split[0]
        self.lines = {}

        file_name_split = line_split[0].split('-')
        file_name_subdir1 = file_name_split[0]
        file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
        file_base_name = line_split[0] + '.png'
        file_name = data_dir / 'images' / "words" / \
            file_name_subdir1 / file_name_subdir2 / file_base_name + ".png"

        text = ' '.join(line_split[8:])

        self.id = file_name_split[3]
        self.form_id = file_name_split[0] + "-" + file_name_split[1]
        self.line_id = file_name_split[2]
        self.full_id = line_split[0]
        self.x = line_split[3]
        self.y = line_split[4]
        self.width = line_split[5]
        self.height = line_split[6]
        self.text = text
        self.file_name = file_name




