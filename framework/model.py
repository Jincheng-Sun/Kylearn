from absl import flags
import tensorflow as tf
from abc import abstractmethod
FLAGS = flags.FLAGS

class Model():
    def __init__(self, ckpt_path, tsboard_path, **kwargs):
        self.checkpoint_path = ckpt_path
        self.tensorboard_path = tsboard_path
        self.session = tf.Session()
        self.global_step = tf.train.get_or_create_global_step()
        # self.saver = tf.train.Saver(max_to_keep=11)
        # self.writer = tf.summary.FileWriter(self.tensorboard_path)

        self.saver = None  # saver should define after creating the graph
        self.writer = None


    @abstractmethod
    def classifier(self, **kwargs):
        pass

    @abstractmethod
    def initialize_variables(self, **kwargs):
        # with tf.get_collection("global_variables"):
        pass

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    def save_checkpoint(self):
        self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)

    def save_tensorboard_graph(self):
        self.writer.add_graph(self.session.graph)
        return self.writer.get_logdir()

    def restore_checkpoint(self, number):
        self.saver.restore(self.session, self.checkpoint_path+'-%s'%str(number))







