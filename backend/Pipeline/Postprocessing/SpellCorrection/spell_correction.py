import re
from collections import Counter
import sys
sys.path.append('Pipeline/utils')
from utils import read_charlist


class SpellCorrection:
    def __init__(self, text_correction_file, charlist) -> None:
        self.all_words = Counter(self.words(open(text_correction_file).read()))
        self.charlist = charlist
        
    def words(self, text):
        """ Preprocess words from file """
        return re.findall(r'\w+', text.lower())

    def P(self, word): 
        """ Probability of `word`."""
        N = sum(self.all_words.values())
        return self.all_words[word] / N

    def candidates(self, word): 
        """ Generate possible spelling corrections for word."""
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        """ The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.all_words)

    def edits1(self, word):
        """ All edits that are one edit away from `word`."""
        letters    = read_charlist(self.charlist)
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        """ All edits that are two edits away from `word`. """
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def correction(self, word): 
        """ Most probable spelling correction for word."""
        isUpper = word[0].isupper()
        word = word.lower()
        corrected_word =  max(self.candidates(word), key=self.P)
        if(isUpper):
            corrected_word = corrected_word[0].upper() + corrected_word[1:]
        return corrected_word