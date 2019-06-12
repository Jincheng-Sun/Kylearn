import numpy as np
import tensorflow as tf
from utils import utils
from framework.model import Model
from utils.string_utils import DictFormatter
from visualization.draw_matrix import *
import collections

class ResnetModel(Model):

    def __init__(self, ckpt_path, tsboard_path, network, input_shape, num_classes, lr, batch_size):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = batch_size
        self.patience = 0

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None] + input_shape, name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None], name='alarm')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')
        with tf.variable_scope('regressor'):
            net = self.classifier(network, self.input_x, num_classes=num_classes,
                                  is_training=self.is_training)
            self.logits = net
            self.loss = tf.losses.mean_squared_error(tf.expand_dims(self.input_y, axis=1), self.logits)

        self.best_loss = 1000

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def classifier(self, network, input, num_classes, is_training):

        return network.network(inputs=input, num_classes=num_classes, is_training=is_training)

    def initialize_variables(self, **kwargs):
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.run(init_op)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def train(self, dataset, lr):
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=round(dataset.train_set.shape[0]/self.batch_size),
                                                       max_step=10000)  # batch*evaluation_span = dataset size = one epoch

        for batch in dataset.training_generator(batch_size=self.batch_size):

            results, loss, _ = self.run([self.logits, self.loss, self.train_op],
                                                    feed_dict={self.input_x: batch['x'],
                                                               self.input_y: batch['y'],
                                                               self.is_training: True})
            step_control = self.run(self.training_control)
            if step_control['time_to_print']:
               print('train_loss= ' + str(loss) + '          round' + str(step_control['step']))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(dataset.eval_set)
                self.save_checkpoint()
                if if_stop:
                    break

    def evaluate(self, eval_data):
        step, results = self.run([self.global_step, self.loss],
                                         feed_dict={self.input_x: eval_data['x'],
                                                    self.input_y: eval_data['y'],
                                                    self.is_training: False})
        print(' val_loss = ' + str(results) + '          round: ' + str(step))
        '''early stoping'''
        loss = results
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == 10:
            stop_training = True
        else:
            stop_training = False


        return stop_training

    # def errors(self, logits, labels):
    #     logits[logits >= 0.5] = 1
    #     logits[logits < 0.5] = 0
    #     labels = tf.cast(labels, tf.int64)
    #     per_sample = tf.to_float(tf.not_equal(logits, labels))
    #     mean = tf.reduce_mean(per_sample)
    #     return mean

    def plot(self, dataset, threshold=0.5):
        results= self.run([self.logits],feed_dict={
                                                                self.input_x: dataset.test_set['x'],
                                                                # self.input_y: dataset.test_set['y'],
                                                                self.is_training: False})

        print(dataset.test_set['y'].shape)
        results = np.array(results).squeeze()
        print(results)
        results[results >= threshold] = 1
        results[results < threshold] = 0

        cm = cm_metrix(dataset.test_set['y'], results)

        cm_analysis(cm, ['Normal', 'malfunction'], precision=True)
