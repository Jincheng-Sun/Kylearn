import numpy as np
import tensorflow as tf
from utils import utils
from framework.model import Model
from utils.string_utils import DictFormatter
from visualization.draw_matrix import *
from evaluation.metrics import confusion_matrix
import collections

class Lstm_model():
    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, feature_num,
                 batch_size, lr, regression=False, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience

        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.float32, [batch_size, timestemps_in, feature_num] + input_shape, name='inputs')
            self.targets = tf.placeholder(tf.float32, [batch_size, timestemps_out, feature_num], name='targets')
            self.target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
            self.max_target_len = tf.reduce_max(self.target_sequence_length, name='max_target_lenth')
            self.source_sequence_length = tf.placeholder(tf.int32, [None], name='source_sequence_length')

            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        def network(inputs, units, num_layers, keep_prob, output_num):
            stacked_cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(units), keep_prob) for _ in
                 range(num_layers)])
            outputs, state = tf.nn.dynamic_rnn(cell=stacked_cells, inputs=inputs, dtype=tf.float32)
            weight = tf.get_variable('weight', [units, output_num], trainable=True)
            bias = tf.get_variable('bias', [1, 1, output_num], trainable=True)
            outputs = tf.einsum('btu,uo->bto', outputs, weight)
            logits = tf.add(outputs, bias)
            outputs = tf.nn.tanh(logits)
            return outputs, state

        with tf.variable_scope('logits'):
            output, _ = network(self.inputs, 128, 2, 0.5, feature_num)







