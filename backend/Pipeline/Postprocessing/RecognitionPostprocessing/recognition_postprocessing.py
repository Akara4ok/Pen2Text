""" Class for text postprocessing """
import sys
import tensorflow as tf
import numpy as np
sys.path.append("Pipeline/Postprocessing")
from postprocessor import Postprocessor
sys.path.append("Pipeline/Postprocessing/RecognitionPostprocessing/SpellCorrection")
from spell_correction import SpellCorrection
sys.path.append('Pipeline/utils')
from utils import simple_decode
sys.path.append('Pipeline/utils')
from utils import read_charlist

class RecognitionPostprocessing(Postprocessor):
    """ Class for text postprocessing """
    def __init__(self, spell_correction_file: str, char_list_path: str) -> None:
        self.spell_correction = SpellCorrection(charlist=char_list_path, text_correction_file=spell_correction_file)
        self.char_list = read_charlist(char_list_path)

    def isPunctuationChar(self, s: str) -> bool:
        """ Check if symbol or string is puctuatuion char """
        if(s == '.' or s == ','):
            return True
        return False
    
    def process_single_text(self, label: str, last_label: str) -> tuple:
        """ Process current label with information about previous one """
        if(label == ''):
            return (None, None)

        char = label[-1]
        if(self.isPunctuationChar(char)):
            label = label[:-1]
            
        if(label.isalpha()):
            label = self.spell_correction.correction(label)
        
        if(self.isPunctuationChar(char)):
            label = label + char

        if(self.isPunctuationChar(label) and last_label == None):
            return (None, None)
        
        if (self.isPunctuationChar(label) and self.isPunctuationChar(last_label[-1])):
            return (None, None)
            
        if (self.isPunctuationChar(label) and not self.isPunctuationChar(last_label[-1])):
            last_label = last_label + label
            return (None, last_label)
        
        return (label, None)
    
    def process(self, x: tf.Tensor) -> np.ndarray:
        result = []
        out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(x, input_length=np.ones(x.shape[0])*x.shape[1],
                                greedy=False)[0][0])

        for i, x in enumerate(out):
            label = simple_decode(x, self.char_list)

            (current_label, prev_label) = self.process_single_text(label, result[-1] if len(result) > 0 else None)

            if(prev_label != None):
                result[-1] = prev_label
            
            if(current_label != None):
                result.append(current_label)
        
        return result