from model.ELECTRA_models import map_model_to_preprocess, map_name_to_handle
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

class ELECTRA_model:
    model = None

    def __init__(self, name: str, n_outputs: int, trainable: bool) -> None:
        encoder = map_name_to_handle[name]
        preprocess = map_model_to_preprocess[name]

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessing_layer = hub.KerasLayer(preprocess)
        encoder_layer = hub.KerasLayer(encoder, trainable=trainable)
        encoder_inputs = preprocessing_layer(text_input)
        outputs = encoder_layer(encoder_inputs)
        net = outputs['pooled_output']
        
        net = layers.Dense(256, activation='relu')(net)
        net = layers.Dropout(0.2)(net)
        net = layers.Dense(128, activation='relu')(net)
        net = layers.Dropout(0.2)(net)

        out = layers.Dense(n_outputs, activation='softmax')(net)

        model = tf.keras.models.Model(text_input, out)
    
        self.model = model

    def get_model(self):
        return self.model
