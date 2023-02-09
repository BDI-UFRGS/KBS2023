from model.BERT_models import map_model_to_preprocess, map_name_to_handle
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

class LSTM_model:
    model = None

    def __init__(self, encoder_layer, input_dim, output_dim, n_outputs: int) -> None:
        rnn_input = tf.keras.Input(shape=(1,), dtype='string')
        rnn_bi_ltsm = encoder_layer(rnn_input)
        rnn_bi_ltsm = layers.Embedding(input_dim=input_dim, output_dim=output_dim)(rnn_bi_ltsm)
        rnn_bi_ltsm = layers.Bidirectional(layers.LSTM(output_dim))(rnn_bi_ltsm)
        
        net = layers.Dense(256, activation='relu')(rnn_bi_ltsm)
        net = layers.Dropout(0.2)(net)
        net = layers.Dense(128, activation='relu')(net)
        net = layers.Dropout(0.2)(net)

        out = layers.Dense(n_outputs, activation='softmax')(net)

        model = tf.keras.models.Model(rnn_input, out)
    
        self.model = model


    def get_model(self):
        return self.model
