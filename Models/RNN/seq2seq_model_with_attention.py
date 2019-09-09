import numpy as np
import tensorflow as tf
from utils import utils
from framework.model import Model
from utils.string_utils import DictFormatter
from visualization.draw_matrix import *
from evaluation.metrics import confusion_matrix
import collections


class Seq2seq_model(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, feature_num,
                 batch_size, lr, regression=False, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        self.encoder_units = 128

        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.float32, [batch_size, None, feature_num] + input_shape, name='inputs')
            self.targets = tf.placeholder(tf.float32, [batch_size, None, feature_num], name='targets')
            self.target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
            self.max_target_len = tf.reduce_max(self.target_sequence_length, name='max_target_lenth')
            self.source_sequence_length = tf.placeholder(tf.int32, [None], name='source_sequence_length')

            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        def encoder(inputs, units, initial_state):
            # inputs shape          [batch size, sequence length, feature_num]
            # outputs shape         [batch size, sequence length, units]
            # hidden state shape    [batch size, units]

            lstm = tf.keras.layers.LSTM(units=units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

            outputs, state = lstm(inputs, initial_state=initial_state)

            return outputs, state

        def decoder(input, units, hidden_state):
            # inputs shape [batch size, 1, encoder_units]
            # concat shape [batch size, 1, feature_num + encoder_units]
            # outputs shape [batch size, sequence length, units]
            # hidden state shape [batch size, units]
            lstm = tf.keras.layers.LSTM(units=units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

            hidden_state_expandim = tf.expand_dims(hidden_state, axis=1)
            concat = tf.concat([input, hidden_state_expandim], axis=-1)

            output, state = lstm(concat)

            return output, state

        def max_length(tensor):
            return max(len(t) for t in tensor)

        def attention_idea2(encoder_outputs, decoder_input):
            with tf.variable_scope('idea1_attention'):
                decoder_input = tf.tile(decoder_input, [1, encoder_outputs.shape[1], 1])
                concat = tf.concat([encoder_outputs, decoder_input], axis=-1)
                score = tf.nn.tanh(tf.layers.dense(inputs=concat, units=1))
                attention_weights = tf.nn.softmax(score, axis=1)
                encoder_outputs_sum = attention_weights * encoder_outputs
                encoder_outputs_sum = tf.reduce_sum(encoder_outputs_sum, axis=1)
            return encoder_outputs_sum, attention_weights

        def attention_bahdanau(encoder_outputs, hidden_state, units):
            with tf.variable_scope('bahadanau_attention'):
                # encoder outputs       [batch size, sequence length, encoder_units]
                # hidden state          [batch size, encoder_units]
                # hidden_state_exp      [batch size, 1, encoder_units]
                # encoder_outputs_sum   [batch size, units]
                # attention_weights     [batch size, sequence length, 1]

                hidden_state_expandim = tf.expand_dims(hidden_state, 1)
                map1 = tf.layers.dense(inputs=hidden_state_expandim, units=units)
                map2 = tf.layers.dense(inputs=encoder_outputs, units=units)
                score = tf.nn.tanh(tf.layers.dense(inputs=map1 + map2, units=1))
                attention_weights = tf.nn.softmax(score, axis=1)
                encoder_outputs_sum = attention_weights * encoder_outputs
                encoder_outputs_sum = tf.reduce_sum(encoder_outputs_sum, axis=1)
            return encoder_outputs_sum, attention_weights

        def attention_bahdanau_modified(encoder_outputs, hidden_state):
            with tf.variable_scope('bahdanau_modified_attention'):
                hidden_state_repeat = tf.tile(hidden_state, [1, encoder_outputs.shape[1], 1])
                concat = tf.concat([hidden_state_repeat, encoder_outputs], axis=-1)
                score = tf.nn.tanh(tf.layers.dense(inputs=concat, units=1))
                attention_weights = tf.nn.softmax(score, axis=1)
                encoder_outputs_sum = attention_weights * encoder_outputs
                encoder_outputs_sum = tf.reduce_sum(encoder_outputs_sum, axis=1)
            return encoder_outputs_sum, attention_weights

        def seq2seq_model(inputs, units, targets):
            encoder_outputs, encoder_state = encoder(inputs=inputs, units=units,
                                                     initial_state=tf.zeros((batch_size, units)))
            encoder_outputs_sum, attention_weights = attention_bahdanau(encoder_outputs=encoder_outputs,
                                                                        hidden_state=encoder_state, units=128)

            decoder_input = tf.zeros([batch_size, 1, feature_num])
            decoder_state = encoder_outputs_sum
            for t in range(0, targets.shape[1]):
                decoder_output, decoder_state = decoder(decoder_input, units=units, hidden_state=decoder_state)
                # stack decoder_output
                decoder_input = tf.expand_dims(targets[:, t], 1)


            return decoder_outputs

        def fully connected()

        # need to finish loss, training
