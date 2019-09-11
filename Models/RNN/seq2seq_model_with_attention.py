import numpy as np
import tensorflow as tf
from utils import utils
from framework.model import Model
from utils.string_utils import DictFormatter
from visualization.draw_matrix import *
from evaluation.metrics import confusion_matrix
import collections
from Models.RNN.encoder_decoder import Encoder, Decoder, Bahdanau_attention, Seq2Seq_with_attention


class Seq2seq_model(Model):

    def __init__(self, ckpt_path, tsboard_path, units, num_classes, feature_num,
                 batch_size, lr, regression=False, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        self.encoder_units = 128

        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.float32, [batch_size, None, feature_num], name='inputs')
            self.targets = tf.placeholder(tf.float32, [batch_size, None, feature_num], name='targets')
            self.labels = tf.placeholder(tf.float32, [None, num_classes], name='alarm')

            self.target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
            self.max_target_len = tf.reduce_max(self.target_sequence_length, name='max_target_lenth')
            self.source_sequence_length = tf.placeholder(tf.int32, [None], name='source_sequence_length')

            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        def max_length(tensor):
            return max(len(t) for t in tensor)

        encoder = Encoder(units)
        decoder = Decoder(units)
        attention = Bahdanau_attention(256)
        self.model = Seq2Seq_with_attention(units=feature_num,
                                            encoder=encoder,
                                            decoder=decoder,
                                            attention=attention,
                                            batch_size=batch_size,
                                            feature_num=feature_num)

        initial_state = None
        encoder_outputs, encoder_state = self.model.encode(self.inputs, initial_state=initial_state)
        self.outputs_training = self.model.decode_training(encoder_outputs, encoder_state=encoder_state,
                                                           targets=self.targets, target_length=10)
        self.outputs_implementing = self.model.teacher_forcing(encoder_outputs, encoder_state=encoder_state,
                                                               target_length=self.target_sequence_length)
        self.classifier = tf.keras.layers.Dense(units=num_classes)
        self.logits_training = self.classifier(self.outputs_training)
        self.logits_implementing = self.classifier(self.outputs_training)

        def error_classification(outputs, targets, labels):
            logits = self.classifier(outputs)
            loss_1 = tf.reduce_mean(tf.square(outputs - targets))
            loss_2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            prediction = tf.nn.softmax(logits)
            prediction = tf.argmax(prediction, 1)
            real = tf.argmax(labels, 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, real), tf.float32))

            return logits, loss_1 + loss_2, prediction, accuracy

        def error_single_loss_classification(outputs, labels):
            logits = self.classifier(outputs)
            loss_2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            prediction = tf.nn.softmax(logits)
            prediction = tf.argmax(prediction, 1)
            real = tf.argmax(labels, 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, real), tf.float32))

            return logits, loss_2, prediction, accuracy

        def error_regression(outputs, targets, labels, threshold=0.9):
            logits = self.classifier(outputs)
            proba = tf.nn.sigmoid(logits)
            loss_1 = tf.reduce_mean(tf.square(outputs - targets))
            loss_2 = tf.reduce_mean(tf.square(proba - labels))

            threshold = tf.constant(threshold)
            condition = tf.greater_equal(proba, threshold)
            prediction = tf.where(condition, tf.ones_like(proba), tf.zeros_like(proba),
                                  name='prediction')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

            return logits, loss_1 + loss_2, prediction, accuracy

        def error_single_loss_regression(outputs, labels, threshold=0.9):
            logits = self.classifier(outputs)
            proba = tf.nn.sigmoid(logits)
            loss_2 = tf.reduce_mean(tf.square(proba - labels))

            threshold = tf.constant(threshold)
            condition = tf.greater_equal(proba, threshold)
            prediction = tf.where(condition, tf.ones_like(proba), tf.zeros_like(proba),
                                  name='prediction')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

            return logits, loss_2, prediction, accuracy

        self.logits, self.loss, self.prediction, self.accuracy = error_classification(self.outputs_training,
                                                                                      self.targets, self.labels)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        '''when training, the moving_mean and moving_variance need to be updated. 
        By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
        so they need to be executed alongside the train_op.
        Also, be sure to add any batch_normalization ops before getting the update_ops collection. 
        Otherwise, update_ops will be empty, and training/inference will not work properly. '''

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=11)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, **kwargs):
        pass

a = Seq2seq_model('test/','test/', 1024, 2, 45, 100, 0.001)