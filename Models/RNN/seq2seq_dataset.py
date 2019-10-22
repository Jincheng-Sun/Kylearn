import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from framework.dataset import Dataset
from utils.mini_batch import random_index
import collections
class Seq2seq_dataset(Dataset):
    def __init__(self, feature_path, target_path, label_path, out_num):
        super().__init__()

        # train set
        features = np.load(feature_path)
        targets = np.load(target_path)
        labels = np.load(label_path)
        # labels = np.load(label_path).reshape(-1)
        # labels = np.eye(out_num)[labels].reshape([-1, out_num])
        # labels = labels.reshape([-1, out_num])

        self.train_set = np.zeros(features.shape[0], dtype=[
            ('inputs', np.float32, (features.shape[1:])),
            ('targets', np.float32, (targets.shape[1:])),
            ('labels', np.int32, (labels.shape[1:]))
        ])
        self.train_set['inputs'] = features
        self.train_set['targets'] = targets
        self.train_set['labels'] = labels

        self.train_set, self.test_set = train_test_split(self.train_set, test_size=0.2, random_state=12)



        self.train_set, self.val_set = train_test_split(self.train_set, test_size=0.002, random_state=22)

    def labeled_pos_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.train_set) > 0
        anomaly = self.train_set[self.train_set['labels'].flatten() == 1]
        for batch_idxs in random_index(len(anomaly), batch_size, random):
            yield anomaly[batch_idxs]

    def labeled_neg_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.train_set) > 0
        normal = self.train_set[self.train_set['labels'].flatten() == 0]
        for batch_idxs in random_index(len(normal), batch_size, random):
            yield normal[batch_idxs]

    def training_generator(self, batch_size):
        labeled_pos = self.labeled_pos_generator(batch_size=round(batch_size/2))
        labeled_neg = self.labeled_neg_generator(batch_size=round(batch_size/2))
        return zip(labeled_pos, labeled_neg)