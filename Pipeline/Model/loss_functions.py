""" Possible used loss function for models """
import tensorflow as tf

def ctc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ Compute the training-time loss value """
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    batch_size = y_pred.get_shape().as_list()[0]
    input_length = input_length * tf.ones(shape=(batch_size, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_size, 1), dtype="int64")
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)