""" Constants for dataloaders, preprocessing and model """

#general
BATCH_SIZE = 10
BATCH_SIZE_SEG = 1

#data loader constants
TRAIN_PERCENT = 0.7
VAL_PERCENT = 0.1
TEST_PERCENT = 0.2
CHAR_DIR = 'Pipeline/CharList.txt'
UKR_CHAR_DIR = 'Pipeline/UkrCharList.txt'
IMG_NUM = 120000
RANDOM_SEED = 42

#preprocessing constants
WIDTH = 128
HEIGHT = 32
DEFAULT_WORD_SEP = 30
DEFAULT_WORD_NUM = 5
CLUSTERING_PERCENT = 0.1

#text constants
MAX_LEN = 22
TEXT_CORRECTION_FILE_ENG = "Data/SpellCorrection/big.txt"
TEXT_CORRECTION_FILE_UKR = "Data/SpellCorrection/big_ukrainian.txt"

#training constants
LINE_MODE = False

