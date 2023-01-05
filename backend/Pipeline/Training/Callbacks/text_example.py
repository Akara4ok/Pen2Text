"""Displays a batch of outputs after every epoch."""
import sys
import numpy as np
import tensorflow as tf
sys.path.append('Pipeline/utils')
from utils import simple_decode
class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset: tf.data, model: tf.Tensor, char_list: list) -> None:
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.char_list = char_list

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """ predict outputs on validation images """
        predictions = []
        targets = []
        dataset = self.dataset.shuffle(
            1000,
            reshuffle_each_iteration=False)

        for batch in dataset:
            X, y = batch
            predictions = self.model.predict(X)
            targets.extend(y.numpy())
            break
        
        # use CTC decoder
        out = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                                greedy=False)[0][0])

        #print(out)
        # i = 0
        for i, x in enumerate(out):
            print("original_text =", simple_decode(targets[i], self.char_list))
            print("predicted text =", simple_decode(x, self.char_list))   
            print('\n')
            if(i == 3):
                break