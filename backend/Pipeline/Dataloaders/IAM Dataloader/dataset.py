"""Classes for representing iam dataset"""

import os
import string

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
        self.x = int(line_split[3])
        self.y = int(line_split[4])
        self.width = int(line_split[5])
        self.height = int(line_split[6])
        self.text = text
        self.file_name = file_name



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

    def add_word(self, id: int, word: Word):
        """ Add word to line by id """
        self.words[id] = word

    def get_box(self) -> tuple:
        """ Get line bboxes """
        return (self.x, self.y, self.width, self.height)

    def get_word_boxes(self) -> list:
        """ Get bboxes of words from line """
        word_array = list(self.words.values())
        # print(len(word_array))
        bounding_boxes = []
        for word in word_array:
            bounding_boxes.append((word.x, word.y, word.width, word.height))
        return bounding_boxes
    
    def get_text(self) -> str:
        """ Get text from line """
        result = ''
        word_array = list(self.words.values())
        punctation = string.punctuation
        for word in word_array:
            if(word.text[:-1] in punctation):
                result += word.text[:-1]
            else:
                result += ' ' + word.text[:-1]
        
        return result

class Form():
    """Class for representing forms"""

    def __init__(self, data_dir: str, line_split: str) -> None:
        self.id = line_split[0]
        self.file_name = data_dir / "images" / "forms" / line_split[0] + ".png"
        self.lines = {}
    
    def add_line(self, id: int, line: Line) -> None:
        """ Add line to form by id """
        self.lines[id] = line

    def add_word(self, line_id: int, word_id: int, word: Word) -> None:
        """ Add word to line in form by id """
        self.lines[line_id].add_word(word_id, word)
    
    def get_line_boxes(self) -> list:
        """ Get bounding box of all lines from form """
        line_array = list(self.lines.values())
        bounding_boxes = []
        for line in line_array:
            bounding_boxes.append((line.x, line.y, line.width, line.height))
        return bounding_boxes
    
    def get_word_boxes(self) -> list:
        """ Get bounding forms of all words from form """
        line_array = list(self.lines.values())
        bounding_boxes = []
        for line in line_array:
            bounding_boxes.extend(line.get_word_boxes())
        return bounding_boxes

    def get_text(self) -> str:
        """ Get text from form """
        line_array = list(self.lines.values())
        result = ''
        for line in line_array:
            result += line.get_text()
        return result[1:]