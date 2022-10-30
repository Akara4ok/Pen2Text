"""Displays a batch of outputs after every epoch."""
import sys
import numpy as np
import tensorflow as tf
sys.path.append('Pipeline/utils')
from utils import decode_batch_predictions

class CallbackEval(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset, model, num_to_char):
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.num_to_char = num_to_char

    def on_epoch_end(self, epoch: int, logs=None):
        print("---------------check--------------")
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = self.model.predict(X, verbose=0)
            
            batch_predictions = decode_batch_predictions(batch_predictions, self.num_to_char)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(self.num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)
