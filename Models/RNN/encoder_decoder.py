import numpy as np
import tensorflow as tf
from utils import utils
from framework.model import Model
from utils.string_utils import DictFormatter
from visualization.draw_matrix import *
from evaluation.metrics import confusion_matrix
import collections


class Encoder():
    def __init__(self, units):
        self.units = units
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, inputs, initial_state=None):
        # inputs shape          [batch size, sequence length, feature_num]
        # initial_state shape   [batch size, units]
        outputs, state = self.gru(inputs, initial_state=initial_state)
        # outputs shape         [batch size, sequence length, units]
        # state shape           [batch size, units]
        return outputs, state


class Decoder():
    def __init__(self, units):
        self.units = units
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, input, hidden_state):
        # input shape           [batch size, 1, feature_num]
        # hidden state shape    [batch size, encoder_units]
        hidden_state_expandim = tf.expand_dims(hidden_state, axis=1)
        # concat shape [batch size, 1, feature_num + encoder_units]
        concat = tf.concat([input, hidden_state_expandim], axis=-1)
        output, state = self.gru(concat)
        # outputs shape         [batch size, 1, units]
        # state shape           [batch size, units]
        return output, state


class Seq2Seq_with_attention():
    def __init__(self, units, encoder, decoder, attention, feature_num):
        self.fc = tf.keras.layers.Dense(units)
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.feature_num = feature_num
        self.units = units

    # def initial_state(self):
    #     initial_state = tf.zeros_like([self.batch_size, self.encoder.units])
    #     return initial_state

    def encode(self, inputs, initial_state):
        # inputs shape          [batch size, sequence length, feature_num]
        encoder_outputs, encoder_state = self.encoder(inputs=inputs, initial_state=initial_state)
        # encoder_outputs shape [batch size, sequence length, units]
        # encoder_state shape   [batch size, units]
        return encoder_outputs, encoder_state

    def decode(self, input, hidden_state, encoder_outputs):
        # input shape           [batch size, 1, feature_num]
        # hidden_state shape    [batch size, units]

        context_vector, _ = self.attention(encoder_outputs, hidden_state)
        decoder_output, decoder_state = self.decoder(input, hidden_state=context_vector)
        # decoder_output shape      [batch size, 1, units]
        # decoder_state shape       [batch size, units]
        decoder_output = self.fc(decoder_output)
        return decoder_output, decoder_state



    def decode_training(self, encoder_outputs, encoder_state, targets, target_length, dec_input_init):
        # targets shape             [batch size, target length, feature_num]
        decoder_outputs, decoder_state = self.decode(dec_input_init, hidden_state=encoder_state,
                                                     encoder_outputs=encoder_outputs)
        for t in range(0, target_length - 1):
            dec_input = tf.expand_dims(targets[:, t], 1)
            decoder_output, decoder_state = self.decode(dec_input, hidden_state=decoder_state,
                                                        encoder_outputs=encoder_outputs)
            # stack decoder_output
            decoder_outputs = tf.concat([decoder_outputs, decoder_output], axis=1)
            decoder_outputs = tf.nn.sigmoid(decoder_outputs)
        return decoder_outputs

    def teacher_forcing(self, encoder_outputs, encoder_state, target_length, dec_input_init):
        decoder_outputs, decoder_state = self.decode(dec_input_init, hidden_state=encoder_state,
                                                     encoder_outputs=encoder_outputs)
        dec_input = decoder_outputs
        for t in range(0, target_length - 1):
            decoder_output, decoder_state = self.decode(dec_input, hidden_state=decoder_state,
                                                        encoder_outputs=encoder_outputs)
            # stack decoder_output
            decoder_outputs = tf.concat([decoder_outputs, decoder_output], axis=1)
            dec_input = decoder_output
        return decoder_outputs


class Bahdanau_attention():
    def __init__(self, units):
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(units)

    def __call__(self, encoder_outputs, hidden_state):
        hidden_state_expandim = tf.expand_dims(hidden_state, 1)
        map1 = self.fc1(hidden_state_expandim)
        map2 = self.fc2(encoder_outputs)
        map = map1 + map2
        score = tf.nn.tanh(tf.layers.dense(inputs=map, units=1))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * encoder_outputs
        context = tf.reduce_sum(context, axis=1)
        # encoder_outputs_sum   [batch size, encoder_units]
        # attention_weights     [batch size, sequence length, 1]
        return context, attention_weights
