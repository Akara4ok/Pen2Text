"""Class for generation ukrainian words"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
import cv2
import string
import random

def read_charlist(file_path: str) -> list:
    """ Read possible char lists from file """
    char_list_file = open(file_path)
    line = char_list_file.readline()
    charlist = [x for x in line]
    return charlist

class UkrainianWordsGen():
    """Class for generation ukrainian words"""

    def __init__(self,
                 word_file: str,
                 char_list: str,
                 folder_path: str,
                 csv_metadata: str) -> None:
        self.word_file = word_file
        self.folder_path = folder_path
        self.csv_metadata = csv_metadata
        self.char_list = read_charlist(char_list)

        data = pd.read_csv(self.folder_path + "/" + self.csv_metadata, sep=',', encoding='utf-8')
        self.chars = {}

        for char in data.label.unique():
            self.chars[char] = []
            self.chars[char.upper()] = []

        for row in data.itertuples():
            path = self.folder_path + "/" + row.filename
            label = row.label
            is_upper = row.is_uppercase
            if(is_upper):
                label = label.upper()
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img[row.top - 20:row.bottom + 20, row.left:row.right]
            self.chars[label].append(img)

            self.words_array = []

            with open(self.word_file, 'r') as f:
                self.words = [word for line in f for word in line.split()]

    def gen_with_saving(self, totalCount: int) -> np.ndarray:
        """ gen total count examples """
        path_array = []

        index = 0
        cur_index = 0
        while (index < totalCount):
            sanitize_word = self.words[cur_index].translate(str.maketrans('', '', string.punctuation))
            
            word_img_path = "words" + "/" + sanitize_word + str(index) + ".png"
            word_img = self.create_word(sanitize_word)
            if(len(word_img) == 0):
                continue
            cur_index += 1
            cv2.imwrite(self.folder_path + "/" + word_img_path, word_img)

            self.words_array.append(sanitize_word)
            path_array.append(word_img_path)

            preprocessed_data = pd.DataFrame({'path': path_array, 'word': self.words_array})
            preprocessed_data.to_csv(self.folder_path + "/" + "words.csv", index=False)

            index += 1

    def gen(self, totalCount: int) -> np.ndarray:
        cur_words_count = 0
        result = []
        while cur_words_count < totalCount:
            word_no = random.randint(0, len(self.words))
            sanitize_word = self.words[word_no].translate(str.maketrans('', '', string.punctuation))
            word_img = self.create_word(sanitize_word)
            if(len(word_img) == 0):
                continue
            
            cur_words_count += 1
            result.append((sanitize_word, word_img))
        
        return result


    def create_word(self, sanitize_word: str) -> np.ndarray:
        """ Generate word picture with letters  """
        words_chars_imgs = []

        if(sanitize_word == ''):
            return []
            
        isCorrectWord = True
        for c in sanitize_word:
            if (c not in self.char_list):
                isCorrectWord = False
                break
        
        if(not isCorrectWord):
            return []

        word_height = -1
        word_width = 0

        for char in list(sanitize_word):
            if(len(self.chars[char]) == 0):
                char = char.lower()
                sanitize_word = sanitize_word.lower()
            char_ind = random.randint(0, len(self.chars[char]) - 1)
            img = self.chars[char][char_ind]
            word_height = max(word_height, int(img.shape[0]))
            word_width += int(img.shape[1])
            _, img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
            words_chars_imgs.append(img)

        word_height += 30
        word_width += 20
        word_img = np.zeros((word_height, word_width))
        start_offset = 5
        x_offset = start_offset
        for char_img in words_chars_imgs:
            y_offset = max(0, (int(start_offset + random.gauss(0, 3))))
            x_offset = max(0, x_offset + int(random.gauss(0, 5)))
            char_img_height = char_img.shape[0]
            char_img_width = char_img.shape[1]
            if(x_offset + char_img_width >= word_width):
                x_offset = word_width - 1 - char_img_width
            

            for y in range(y_offset, y_offset+char_img_height):
                for x in range(x_offset, x_offset+char_img_width):
                    word_img[y, x] = max(word_img[y, x], char_img[y - y_offset, x - x_offset])
            x_offset += char_img_width

        _, word_img = cv2.threshold(word_img, 127, 255, cv2.THRESH_BINARY_INV)

        return word_img

# ukr_words = UkrainianWordsGen('./Data/SpellCorrection/big_ukrainian.txt', "Pipeline/UkrCharList.txt", './Data/Ukrainian Characters', 'glyphs.csv')
# ukr_words.gen(700)

ukr_words = UkrainianWordsGen('./Data/SpellCorrection/big_ukrainian.txt', "Pipeline/Charlists/Ukr/CharList.txt", './Data/Ukrainian Characters', 'glyphs.csv')
words = ukr_words.gen(7)

for text, img in words:
    print(text)
    cv2.imshow("img", img)
    cv2.waitKey(0)