import sys
import tensorflow as tf
sys.path.append('Pipeline/Model')
from improved_ocr import ImprovedPen2Text
from loss_functions import ctc_loss
sys.path.append('Pipeline/utils')
from utils import read_charlist
sys.path.append('Pipeline/')
import model_settings as settings

char_list = read_charlist("./Pipeline/Charlists/Eng/CharListLettersNumbers.txt")
max_len = settings.MAX_LEN

model=ImprovedPen2Text(char_list)
model.compile(loss=ctc_loss, optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001))
model.build(input_shape=(1,32,128, 1))
print(model.summary())
