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

        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.float32, [batch_size, None, feature_num] + input_shape, name='inputs')
            self.targets = tf.placeholder(tf.float32, [batch_size, None, feature_num], name='targets')
            self.target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
            self.max_target_len = tf.reduce_max(self.target_sequence_length, name='max_target_lenth')
            self.source_sequence_length = tf.placeholder(tf.int32, [None], name='source_sequence_length')

            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        def encoder(inputs, units, num_layers, keep_prob):
            stacked_cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(units), keep_prob) for _ in
                 range(num_layers)])
            outputs, state = tf.nn.dynamic_rnn(cell=stacked_cells, inputs=inputs, dtype=tf.float32)
            return outputs, state

        def decoder(decoder_input, units, encoder_state, num_layers, output_num, start_tolken,
                    target_sequence_length, max_target_sequence_length, keep_prob):
            stacked_cells = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(units), keep_prob) for _ in
                 range(num_layers)])

            # 3. Output全连接层
            output_layer = tf.layers.Dense(output_num)

            # 4. Training decoder
            with tf.variable_scope("decode"):
                # 得到help对象
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input,
                                                                    sequence_length=target_sequence_length,
                                                                    time_major=False)
                # 构造decoder
                training_decoder = tf.contrib.seq2seq.BasicDecoder(stacked_cells,
                                                                   training_helper,
                                                                   encoder_state,
                                                                   output_layer)

                training_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=max_target_sequence_length)
            # 5. Predicting decoder
            # 与training共享参数
            with tf.variable_scope("decode", reuse=tf.AUTO_REUSE):
                start_tokens = tf.tile(tf.constant([start_tolken], dtype=tf.int32), [batch_size, 1, output_num],
                                       name='start_tokens')

                predicting_helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=lambda outputs: outputs,
                                                                       sample_shape=[output_num],  # again because dim=1
                                                                       sample_dtype=tf.float32,
                                                                       start_inputs=start_tokens,
                                                                       end_fn=lambda sample_ids: False)

                predicting_decoder = tf.contrib.seq2seq.BasicDecoder(stacked_cells,
                                                                     predicting_helper,
                                                                     encoder_state,
                                                                     output_layer)
                predicting_decoder_output, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=max_target_sequence_length)

            return training_decoder_output, predicting_decoder_output

        def seq2seq_model(input_data, targets, output_num, keep_prob, target_sequence_length,
                          max_target_sequence_length, units, num_layers):
            # 获取encoder的状态输出
            _, encoder_state = encoder(input_data, units=units, num_layers=num_layers, keep_prob=keep_prob)

            # 将状态向量与输入传递给decoder
            training_decoder_output, predicting_decoder_output = decoder(decoder_input=targets, units=units,
                                                                         encoder_state=encoder_state,
                                                                         num_layers=num_layers,
                                                                         output_num=output_num, start_tolken=-100,
                                                                         keep_prob=keep_prob,
                                                                         target_sequence_length=target_sequence_length,
                                                                         max_target_sequence_length=max_target_sequence_length
                                                                         )

            return training_decoder_output, predicting_decoder_output





