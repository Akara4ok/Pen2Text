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
        # CNN layer
        self.conv_1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation='selu', padding='same')
        self.pool_1=tf.keras.layers.MaxPool2D(pool_size = (2, 2))

        self.conv_2=tf.keras.layers.Conv2D(
            64, (3, 3), activation = 'selu', padding = 'same')
        self.pool_2=tf.keras.layers.MaxPool2D(pool_size = (2, 2))

        self.conv_3=tf.keras.layers.Conv2D(
            128, (3, 3), activation = 'selu', padding = 'same')

        self.conv_4=tf.keras.layers.Conv2D(
            128, (3, 3), activation = 'selu', padding = 'same')
        self.pool_4=tf.keras.layers.MaxPool2D(pool_size = (2, 1))

        self.conv_5=tf.keras.layers.Conv2D(
            256, (3, 3), activation = 'selu', padding = 'same')
        self.batch_norm_5=tf.keras.layers.BatchNormalization()

        self.conv_6=tf.keras.layers.Conv2D(
            256, (3, 3), activation = 'selu', padding = 'same')
        self.batch_norm_6=tf.keras.layers.BatchNormalization()
        self.pool_6=tf.keras.layers.MaxPool2D(pool_size = (2, 1))

        self.conv_7=tf.keras.layers.Conv2D(64, (2, 2), activation = 'selu')


        # squeezed layer
        self.squeezed=tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.squeeze(x, 1))


        # bidirectional LSTM layers with units=128
        num_units = 128
        self.blstm_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(num_units, return_sequences=True))
        self.blstm_2=tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(num_units, return_sequences=True))


        # output_layer
        self.outputs=tf.keras.layers.Dense(
            len(char_list)+1, activation = 'softmax')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """ Model prediction """

        current_layer=inputs
        current_layer=self.conv_1(current_layer)
        current_layer=self.pool_1(current_layer)

        current_layer=self.conv_2(current_layer)
        current_layer=self.pool_2(current_layer)

        current_layer=self.conv_3(current_layer)

        current_layer=self.conv_4(current_layer)
        current_layer=self.pool_4(current_layer)

        current_layer=self.conv_5(current_layer)
        current_layer=self.batch_norm_5(current_layer)

        current_layer=self.conv_6(current_layer)
        current_layer=self.batch_norm_6(current_layer)
        current_layer=self.pool_6(current_layer)

        current_layer=self.conv_7(current_layer)
    
        current_layer=self.squeezed(current_layer)

        current_layer=self.blstm_1(current_layer)
        current_layer=self.blstm_2(current_layer)

        current_layer=self.outputs(current_layer)

        return current_layer