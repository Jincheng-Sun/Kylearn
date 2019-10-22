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
                 batch_size, lr, target_length, regression=False, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        self.encoder_units = 128
        self.best_loss = 100000


        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.float32, [None, None, feature_num], name='inputs')
            self.targets = tf.placeholder(tf.float32, [None, target_length, feature_num], name='targets')
            self.labels = tf.placeholder(tf.float32, [None, target_length, num_classes], name='alarm')
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
                                            feature_num=feature_num)

        initial_state = None
        encoder_outputs, encoder_state = self.model.encode(self.inputs, initial_state=initial_state)

        dec_input_init = tf.expand_dims(tf.zeros_like(self.targets[:, 0]), axis=1)

        self.outputs_training = self.model.decode_training(encoder_outputs, encoder_state=encoder_state,
                                                           targets=self.targets, target_length=target_length,
                                                           dec_input_init=dec_input_init)
        self.outputs_implementing = self.model.teacher_forcing(encoder_outputs, encoder_state=encoder_state,
                                                               target_length=target_length, dec_input_init=dec_input_init)
        self.classifier = tf.keras.layers.Dense(units=num_classes)

        self.logits_training = self.classifier(self.outputs_training)
        self.logits_implementing = self.classifier(self.outputs_training)

        def error_classification(outputs, targets, labels):
            logits = self.classifier(outputs)
            loss_1 = tf.reduce_mean(tf.square(outputs - tf.nn.sigmoid(targets)))
            loss_2 = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits,
                                                     pos_weight=1000))
            # loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            prediction = tf.nn.softmax(logits)
            prediction = tf.argmax(prediction, 2)
            real = tf.argmax(labels, 2)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, real), tf.float32))

            return logits, loss_1, loss_2, prediction, accuracy

        def error_single_loss_classification(outputs, labels):
            logits = self.classifier(outputs)
            loss_2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            prediction = tf.nn.softmax(logits)
            prediction = tf.argmax(prediction, 2)
            real = tf.argmax(labels, 2)
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

        self.logits, self.loss_1, self.loss_2, self.prediction, self.accuracy = error_classification(self.outputs_training,
                                                                                      self.targets, self.labels)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        '''when training, the moving_mean and moving_variance need to be updated. 
        By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
        so they need to be executed alongside the train_op.
        Also, be sure to add any batch_normalization ops before getting the update_ops collection. 
        Otherwise, update_ops will be empty, and training/inference will not work properly. '''

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss_1+self.loss_2, global_step=self.global_step)
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=11)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=100,
                                                       max_step=100000)  # batch*evaluation_span = dataset size = one epoch

        for batch in dataset.training_generator(batch_size=self.batch_size):

            accuracy, loss_1, loss_2, _ = self.run([self.accuracy, self.loss_1, self.loss_2, self.train_op],
                                         feed_dict={self.inputs: batch['inputs'],
                                                    self.targets: batch['targets'],
                                                    self.labels: batch['labels'],
                                                    self.is_training: True})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
                print('regression_loss= ' + str(loss_1) + 'classification_loss= ' + str(loss_2) + '    train_acc= ' + str(accuracy) + '          round' + str(
                    step_control['step']))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(dataset.val_set)
                self.save_checkpoint()
                if if_stop:
                    break

    def evaluate(self, val_data):
        step, loss_1, loss_2, accuracy = self.run([self.global_step, self.loss_1, self.loss_2, self.accuracy],
                                 feed_dict={self.inputs: val_data['inputs'],
                                            self.targets: val_data['targets'],
                                            self.labels: val_data['labels'],
                                            self.is_training: False})

        print('regression_loss= ' + str(loss_1) + 'classification_loss= ' + str(loss_2) + '    train_acc= ' + str(
            accuracy) + '          round' + str(step))

        '''early stoping'''
        if loss_2 < self.best_loss:
            self.best_loss = loss_2
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == self.patience_max:
            stop_training = True
        else:
            stop_training = False

        return stop_training

    def get_prediction(self, data, is_training=False):
        prediction = self.run(self.prediction, feed_dict={
            self.inputs: data['inputs'],
            self.is_training: is_training
        })
        return prediction

    # def get_accuracy(self, data, is_training=False):
    #     accuracy = self.run(self.accuracy, feed_dict = {
    #         self.input_x: data['x'],
    #         self.input_dev: data['dev'],
    #         self.input_y: data['y'],
    #         self.is_training: is_training
    #     })
    #     return accuracy

    # def get_logits(self, data, is_training=False):
    #     logits = self.run([self.logits], feed_dict={
    #         self.inputs: data['inputs'],
    #         self.targets: val_data['targets'],
    #         self.labels: val_data['labels'],
    #         self.is_training: is_training
    #     })
    #     return logits

    # def get_proba(self, data, is_training=False):
    #     proba = self.run(self.proba, feed_dict={
    #         self.input_x: data['x'],
    #         self.input_dev: data['dev'],
    #         self.is_training: is_training
    #     })
    #     return proba



class Seq2seq_model_2(Model):

    def __init__(self, ckpt_path, tsboard_path, units, num_classes, feature_num,
                 batch_size, lr, target_length, regression=False, threshold=0.99, patience=10):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0
        self.patience_max = patience
        self.encoder_units = 128
        self.best_loss = 100000


        with tf.variable_scope("input"):
            # 0 find a way to make target_length None
            self.inputs = tf.placeholder(tf.float32, [None, None, feature_num], name='inputs')
            self.targets = tf.placeholder(tf.float32, [None, target_length, feature_num], name='targets')
            self.labels = tf.placeholder(tf.float32, [None, num_classes], name='alarm')

            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        encoder = Encoder(units)
        decoder = Decoder(units)
        attention = Bahdanau_attention(256)
        self.model = Seq2Seq_with_attention(units=feature_num,
                                            encoder=encoder,
                                            decoder=decoder,
                                            attention=attention,
                                            feature_num=feature_num)

        initial_state = None
        encoder_outputs, encoder_state = self.model.encode(self.inputs, initial_state=initial_state)

        self.context_vector, self.attention_weight = self.model.attention(encoder_outputs, encoder_state)



        self.classifier = tf.keras.layers.Dense(units=num_classes)


        self.logits = self.classifier(self.context_vector)
        self.proba = tf.nn.sigmoid(self.logits)

        self.loss = tf.nn.weighted_cross_entropy_with_logits(targets=self.labels,logits=self.logits, pos_weight=1)
        self.loss = tf.reduce_mean(self.loss)

        threshold = tf.constant(threshold)

        condition = tf.greater_equal(self.proba, threshold)
        self.prediction = tf.where(condition, tf.ones_like(self.proba), tf.zeros_like(self.proba),
                              name='prediction')
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.labels), tf.float32))

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        '''when training, the moving_mean and moving_variance need to be updated. 
        By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
        so they need to be executed alongside the train_op.
        Also, be sure to add any batch_normalization ops before getting the update_ops collection. 
        Otherwise, update_ops will be empty, and training/inference will not work properly. '''

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.train_op = tf.group([train_op, update_ops])
        self.saver = tf.train.Saver(max_to_keep=self.patience_max+1)
        self.writer = tf.summary.FileWriter(self.tensorboard_path)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=100,
                                                       max_step=100000)  # batch*evaluation_span = dataset size = one epoch

        for pos, neg in dataset.training_generator(batch_size=self.batch_size):

            batch = np.concatenate([pos, neg], axis=0)
            accuracy, loss, _ = self.run([self.accuracy, self.loss, self.train_op],
                                         feed_dict={self.inputs: batch['inputs'],
                                                    self.targets: batch['targets'],
                                                    self.labels: batch['labels'],
                                                    self.is_training: True})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
                print('train_loss= ' + str(loss) + '    train_acc= ' + str(accuracy) + '          round' + str(
                    step_control['step']))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(dataset.val_set)
                self.save_checkpoint()
                if if_stop:
                    break

    def evaluate(self, val_data):
        step, loss, accuracy = self.run([self.global_step, self.loss, self.accuracy],
                                 feed_dict={self.inputs: val_data['inputs'],
                                            self.targets: val_data['targets'],
                                            self.labels: val_data['labels'],
                                            self.is_training: False})

        print('eval_loss= ' + str(loss)  + '    train_acc= ' + str(
            accuracy) + '          round' + str(step))

        '''early stoping'''
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == self.patience_max:
            stop_training = True
        else:
            stop_training = False

        return stop_training
        # 3 finish this part

    def get_prediction(self, data, is_training=False):
        prediction = self.run(self.prediction, feed_dict={
            self.inputs: data['inputs'],
            self.targets: data['targets'],
            self.labels: data['labels'],
            self.is_training: is_training
        })
        return prediction

    def get_accuracy(self, data, is_training=False):
        accuracy = self.run(self.accuracy, feed_dict = {
            self.inputs: data['inputs'],
            self.labels: data['labels'],
            self.is_training: is_training
        })
        return accuracy

    def get_logits(self, data, is_training=False):
        logits = self.run([self.logits], feed_dict={
            self.inputs: data['inputs'],
            self.is_training: is_training
        })
        return logits

    def get_proba(self, data, is_training=False):
        proba = self.run(self.proba, feed_dict={
            self.inputs: data['inputs'],
            self.is_training: is_training
        })
        return proba

    def get_attention_weight(self, data, is_training=False):
        attention = self.run(self.attention_weight, feed_dict={
            self.inputs: data['inputs'],
            self.is_training: is_training
        })
        return attention

