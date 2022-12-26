""" Possible used loss function for models """
import tensorflow as tf

def ctc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Compute the training-time loss value """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)