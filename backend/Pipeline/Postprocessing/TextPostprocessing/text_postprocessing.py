""" Class for text postprocessing """
import sys
sys.path.append('Pipeline/Postprocessing/SpellCorrection')
from spell_correction import SpellCorrection

class TextPostProcessing():
    """ Class for text postprocessing """
    def __init__(self, spell_correction_file: str, char_list_path: str) -> None:
        self.spell_correction = SpellCorrection(charlist=char_list_path, text_correction_file=spell_correction_file)

    def isPunctuationChar(self, s: str) -> bool:
        if(s == '.' or s == ','):
            return True
        return False
    
    def process(self, label: str, last_label: str) -> tuple:
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