"""Class for generation ukrainian words"""

from typing import Tuple
import numpy as np
import pandas as pd
import cv2
import string
import random

def read_charlist(file_path: str):
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

    def gen(self, totalCount: int) -> np.ndarray:
        """ gen total count examples """
        data = pd.read_csv(self.folder_path + "/" + self.csv_metadata, sep=',', encoding='utf-8')
        words_array = []
        path_array = []
        chars = {}

        for char in data.label.unique():
            chars[char] = []
            chars[char.upper()] = []

        for row in data.itertuples():
            path = self.folder_path + "/" + row.filename
            label = row.label
            is_upper = row.is_uppercase
            if(is_upper):
                label = label.upper()
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img[row.top - 20:row.bottom + 20, row.left:row.right]
            chars[label].append(img)

        with open(self.word_file, 'r') as f:
            words = [word for line in f for word in line.split()]

        index = 0
        cur_index = 0
        while (index < totalCount):
            words_chars_imgs = []
            sanitize_word = words[cur_index].translate(str.maketrans('', '', string.punctuation))
            cur_index += 1
            if(sanitize_word == ''):
                continue
            
            isCorrectWord = True
            for c in sanitize_word:
                if (c not in self.char_list):
                    isCorrectWord = False
                    break
            
            if(not isCorrectWord):
                continue

            word_height = -1
            word_width = 0

            for char in list(sanitize_word):
                if(len(chars[char]) == 0):
                    char = char.lower()
                    sanitize_word = sanitize_word.lower()
                char_ind = random.randint(0, len(chars[char]) - 1)
                img = chars[char][char_ind]
                word_height = max(word_height, int(img.shape[0]))
                word_width += int(img.shape[1])
                _, img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
                words_chars_imgs.append(img)

            word_height += 30
            word_width += 20
            word_img = np.zeros((word_height, word_width))
            start_offset = 5
            x_offset = start_offset
            print(f"[{index}] {sanitize_word}")
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
            word_img_path = "words" + "/" + sanitize_word + str(index) + ".png"
            cv2.imwrite(word_img_path, word_img)

            words_array.append(sanitize_word)
            path_array.append(word_img_path)

            preprocessed_data = pd.DataFrame({'path': path_array, 'word': words_array})
            preprocessed_data.to_csv(self.folder_path + "/" + "words.csv", index=False)

            index += 1


        return data


ukr_words = UkrainianWordsGen('./Data/SpellCorrection/big_ukrainian.txt', "Pipeline/UkrCharList.txt", './Data/Ukrainian Characters', 'glyphs.csv')
ukr_words.gen(100000)
        
