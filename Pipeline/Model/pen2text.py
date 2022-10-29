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

    def __init__(self, char_list):
        super().__init__()
        self.char_list = char_list
        # CNN layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (2, 2), (2, 1), (2, 1)]
        self.num_cnn_layers = len(feature_vals)

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
        self.outputs = tf.keras.layers.Dense(len(char_list)+1, activation = 'softmax')

    def call(self, inputs):
        """ Model prediction """
        x = inputs
        for i in range(self.num_cnn_layers):
            x = self.cnn_layers[i](x)
            x = self.pool_layers[i](x)
        x = self.squeezed(x)
        x = self.bidirectional_rnn_1(x)
        x = self.bidirectional_rnn_2(x)
        x = self.outputs(x)
        return x