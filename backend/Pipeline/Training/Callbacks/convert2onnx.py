"""Converting best model to onnx."""
import os
import subprocess
import tensorflow as tf

class ConvertCallback(tf.keras.callbacks.Callback):
    """Converting best model to onnx."""

    def __init__(self, tf_path, onnx_path):
        super().__init__()
        self.tf_path = tf_path
        self.onnx_path = onnx_path
        self.best_val = 10000

    def on_epoch_end(self, epoch: int, logs=None):
        """Converting best model to onnx."""
        if not os.path.exists(os.path.dirname(self.onnx_path)):
            os.makedirs(os.path.dirname(self.onnx_path))
        if logs.get("val_loss") < self.best_val:
            self.best_val = logs.get("val_loss")
            print("Val loss improved, converting model to", self.onnx_path)
            subprocess.run(["python3", "-m", "tf2onnx.convert", "--saved-model",
                           str(self.tf_path), "--output", str(self.onnx_path)])
