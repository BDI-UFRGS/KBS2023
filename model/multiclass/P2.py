from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

class P2_model:
    model = None

    def __init__(self, encoder_layer, input_dim, output_dim, n_outputs: int) -> None:
        rnn_input = tf.keras.Input(shape=(1,), dtype='string')
        rnn_bi_ltsm = encoder_layer(rnn_input)
        rnn_bi_ltsm = layers.Embedding(input_dim=input_dim, output_dim=output_dim)(rnn_bi_ltsm)
        rnn_bi_ltsm = layers.Bidirectional(layers.LSTM(512))(rnn_bi_ltsm)
        rnn_bi_ltsm = layers.Dense(256, activation='relu')(rnn_bi_ltsm)
        rnn_bi_ltsm = layers.Dropout(0.2)(rnn_bi_ltsm)
        rnn_bi_ltsm = layers.Dense(128, activation='relu')(rnn_bi_ltsm)
        rnn_output = layers.Dropout(0.2)(rnn_bi_ltsm)



        fnn_input = tf.keras.Input(shape=(300,))
        fnn_output = layers.Dense(128, activation='relu')(fnn_input)

        merge_layer = tf.keras.layers.Concatenate()([rnn_output, fnn_output])

        out = layers.Dense(n_outputs, activation='softmax')(merge_layer)

        model = tf.keras.models.Model([rnn_input, fnn_input], out)
    
        self.model = model


    def get_model(self):
        return self.model
