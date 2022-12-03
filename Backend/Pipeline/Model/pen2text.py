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
        #CNN layers
        self.conv_1 = tf.keras.layers.Conv2D(
            32, (5, 5), activation='relu', padding='same')
        self.batch_norm_1=tf.keras.layers.BatchNormalization()
        self.pool_1=tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))

        self.conv_2=tf.keras.layers.Conv2D(
            64, (5, 5), activation = 'relu', padding = 'same')
        self.batch_norm_2=tf.keras.layers.BatchNormalization()
        self.pool_2=tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2))

        self.conv_3=tf.keras.layers.Conv2D(
            128, (3, 3), activation = 'relu', padding = 'same')
        self.batch_norm_3=tf.keras.layers.BatchNormalization()
        self.pool_3=tf.keras.layers.MaxPool2D(pool_size = (2, 1), strides = (2, 1))

        self.conv_4=tf.keras.layers.Conv2D(
            128, (3, 3), activation = 'relu', padding = 'same')
        self.batch_norm_4=tf.keras.layers.BatchNormalization()
        self.pool_4=tf.keras.layers.MaxPool2D(pool_size = (2, 1), strides = (2, 1))

        self.conv_5=tf.keras.layers.Conv2D(
            256, (3, 3), activation = 'relu', padding = 'same')
        self.pool_5=tf.keras.layers.MaxPool2D(pool_size = (2, 1), strides = (2, 1))

        # squeeze
        self.squeezed = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = 1))
        
        # RNN layers
        num_units = 128
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Model prediction """

        current_layer=inputs
        current_layer=self.conv_1(current_layer)
        current_layer=self.batch_norm_1(current_layer)
        current_layer=self.pool_1(current_layer)

        current_layer=self.conv_2(current_layer)
        current_layer=self.batch_norm_2(current_layer)
        current_layer=self.pool_2(current_layer)

        current_layer=self.conv_3(current_layer)
        current_layer=self.batch_norm_3(current_layer)
        current_layer=self.pool_3(current_layer)

        current_layer=self.conv_4(current_layer)
        current_layer=self.batch_norm_4(current_layer)
        current_layer=self.pool_4(current_layer)

        current_layer=self.conv_5(current_layer)
        current_layer=self.pool_5(current_layer)

        current_layer=self.squeezed(current_layer)

        current_layer=self.bidirectional_rnn_1(current_layer)
        current_layer=self.bidirectional_rnn_2(current_layer)

        current_layer=self.outputs(current_layer)

        return current_layer
