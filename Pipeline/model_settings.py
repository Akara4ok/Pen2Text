""" Constants for dataloaders, preprocessing and model """

#general
BATCH_SIZE = 10

#data loader constants
TRAIN_PERCENT = 0.7
VAL_PERCENT = 0.1
TEST_PERCENT = 0.2
CHAR_DIR = 'Pipeline/CharList.txt'
IMG_NUM = 1200

#preprocessing constants
WIDTH = 128
HEIGHT = 32
DEFAULT_WORD_SEP = 30
DEFAULT_WORD_NUM = 5
CLUSTERING_PERCENT = 0.1

#text constants
MAX_LEN = 22

#training constants
LINE_MODE = False
