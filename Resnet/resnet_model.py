import numpy as np
import tensorflow as tf
from Framework import utils
from Framework.model import Model
from Framework.string_utils import DictFormatter
from Toolkit.draw import *
import logging

logging.basicConfig(level=logging.INFO, filename='train_log', filemode='a')
LOG = logging.getLogger('train_log')
LOG.setLevel(logging.INFO)


class ResnetModel(Model):

    def __init__(self, ckpt_path, tsboard_path, network, num_classes):
        super().__init__(ckpt_path, tsboard_path)

        self.batch_size = 100
        self.patience = 0

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [None, 3, 86, 1], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None], name='alarm')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        net = self.classifier(network, self.input_x, num_classes=num_classes, scope='classification',
                              is_training=self.is_training)
        net = tf.layers.batch_normalization(inputs=net, training=self.is_training, momentum=0.999)
        self.logits = tf.nn.tanh(net)
        self.loss = tf.losses.huber_loss(self.input_y, self.logits)
        self.loss = tf.reduce_sum(self.loss)
        self.error = self.errors(self.logits, self.input_y)
        self.train_op = None
        self.best_loss = 1000
        self.training_control = utils.training_control(self.global_step, print_span=10,
                                                       evaluation_span=60,
                                                       max_step=100000)  # batch*evaluation_span = dataset size = one epoch
        self.training_metrics = {
            "train/error": self.error,
            "train/loss": self.loss
        }
        self.evaluate_metrics = {
            "eval/error": self.error,
            "eval/loss": self.loss
        }

        self.result_formatter = DictFormatter(
            order=["error", "loss"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.1%}')

    def classifier(self, network, input, num_classes, scope, is_training):

        return network.network(inputs=input, num_classes=num_classes, scope=scope, is_training=is_training)

    def initialize_variables(self, **kwargs):
        self.session.run(tf.global_variables_initializer())

    def train(self, dataset, lr):

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.initialize_variables()

        merged = tf.summary.merge_all()
        LOG.info("Model variables initialized")
        for batch in dataset.training_generator(batch_size=self.batch_size):
            results, _ = self.session.run([self.training_metrics, self.train_op],
                                                    feed_dict={self.input_x: batch['x'],
                                                               self.input_y: batch['y'],
                                                               self.is_training: True})
            i = self.session.run([self.global_step])
            # self.writer.add_summary(merge, i)
            step_control = self.session.run(self.training_control)
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(dataset.eval_set)
                self.save_checkpoint()
                if if_stop:
                    LOG.info('Stop Training')
                    break

    def evaluate(self, eval_data):

        step, results = self.session.run([self.global_step, self.evaluate_metrics],
                                         feed_dict={self.input_x: eval_data['x'],
                                                    self.input_y: eval_data['y'],
                                                    self.is_training: False})
        '''early stoping'''
        loss = results["eval/loss"]
        if loss <= self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == 10:
            stop_training = True
        else:
            stop_training = False

        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))

        return stop_training

    def errors(self, logits, labels):
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample)
        return mean

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def plot(self, dataset, checkpoint):
        self.restore_checkpoint(checkpoint)
        results= self.session.run([self.logits],feed_dict={
                                                                self.input_x: dataset.test_set['x'],
                                                                # self.input_y: dataset.test_set['y'],
                                                                self.is_training: False})

        print(dataset.test_set['y'].shape)
        results = np.squeeze(np.argmax(results, axis=-1))
        print(results.shape)
        cm = cm_metrix(dataset.test_set['y'], results)

        cm_analysis(cm, ['Normal', 'malfunction'], precision=True)
