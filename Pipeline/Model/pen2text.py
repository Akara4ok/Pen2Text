"""
Model Pen2Text which consists with
CNN:
5 cnn layers. All layers has three operations:
    1. CNN layer with filter size 5x5 for first two layer and 3x3 for the last three layers
    2. Activation function as Relu
    3. Pooling layer

RNN:
2 rnn layers(Bidirectional LSTM version)

CTC:
loss function
Gets RNN matrix, decodes it and returns text
"""

import tensorflow as tf

class Pen2Text(tf.keras.Model):
    """ Model architecture class """

    def __init__(self, char_list: list):
        super().__init__()
        self.char_list = char_list
        # CNN layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (2, 2), (2, 1), (2, 1)]
        self.num_cnn_layers = len(feature_vals)

        self.cast_input = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))
        self.expand_lambda = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 3))

        self.cnn_layers = []
        self.pool_layers = []
        for i in range(self.num_cnn_layers):
            self.cnn_layers.append(
                tf.keras.layers.Conv2D(
                    filters=feature_vals[i],
                    kernel_size=kernel_vals[i],
                    padding="same",
                    activation="relu"
                )
            )

            self.pool_layers.append(
                tf.keras.layers.MaxPooling2D(
                    pool_size=pool_vals[i],
                    strides=stride_vals[i],
                    padding='VALID'
                )
            )

        # squeeze
        self.squeezed = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = 1))
        
        # RNN layers
        num_units = 256
        lstm_layer = tf.keras.layers.LSTM(
            num_units,
            return_sequences=True,
            dropout=0.2
        )
        self.bidirectional_rnn_1 = tf.keras.layers.Bidirectional(
            layer=lstm_layer
        )
        self.bidirectional_rnn_2 = tf.keras.layers.Bidirectional(
            layer=lstm_layer
        )

        #output
        self.outputs = tf.keras.layers.Dense(len(char_list)+2, activation = 'softmax')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Model prediction """

        current_layer = inputs
        current_layer = self.cast_input(current_layer)
        current_layer = self.expand_lambda(current_layer)
        for i in range(self.num_cnn_layers):
            current_layer = self.cnn_layers[i](current_layer)
            current_layer = self.pool_layers[i](current_layer)
        current_layer = self.squeezed(current_layer)
        current_layer = self.bidirectional_rnn_1(current_layer)
        current_layer = self.bidirectional_rnn_2(current_layer)
        current_layer = self.outputs(current_layer)
        return current_layer
