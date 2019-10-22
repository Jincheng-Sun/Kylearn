import numpy as np
from itertools import islice, chain
from framework.dataset import Dataset
from sklearn.model_selection import train_test_split
from utils.mini_batch import random_index
class Mean_Teacher_dataset(Dataset):
    def __init__(self, feature_path, dev_path, label_path,
                 unlabeled_feature_path, unlabeled_dev_path):
        super().__init__()
        # train set
        X = np.load(feature_path + '_train.npy')
        dev = np.load(dev_path + '_train.npy')
        y = np.load(label_path + '_train.npy')

        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.train_set['x'] = X
        self.train_set['dev'] = dev
        self.train_set['y'] = y

        # unlabeled train set
        X_un = np.load(unlabeled_feature_path + '_train.npy')
        dev_un = np.load(unlabeled_dev_path + '_train.npy')
        assert X_un.shape[0] == dev_un.shape[0]

        self.unlabeled_train_set = np.zeros(X_un.shape[0], dtype=[
            ('x', np.float32, (X_un.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.unlabeled_train_set['x'] = X_un
        self.unlabeled_train_set['dev'] = dev_un
        self.unlabeled_train_set['y'] = -1

        # test_set
        X2 = np.load(feature_path + '_test.npy')
        dev2 = np.load(dev_path + '_test.npy')
        y2 = np.load(label_path + '_test.npy')
        assert X2.shape[0] == y2.shape[0]
        assert dev2.shape[0] == y2.shape[0]

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.test_set['x'] = X2
        self.test_set['dev'] = dev2
        self.test_set['y'] = y2

        self.test_set, self.val_set = train_test_split(self.test_set, test_size=0.05, random_state=22)

    def labeled_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.train_set) > 0
        for batch_idxs in random_index(len(self.train_set), batch_size, random):
            yield self.train_set[batch_idxs]

    def unlabeled_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.unlabeled_train_set) > 0
        for batch_idxs in random_index(len(self.unlabeled_train_set), batch_size, random):
            yield self.unlabeled_train_set[batch_idxs]

    def training_generator(self, batch_size=100, portion = 0.5):
        labeled = self.labeled_generator(batch_size=batch_size - int(batch_size*portion))
        unlabeled = self.unlabeled_generator(batch_size = int(batch_size*portion))
        return zip(labeled, unlabeled)

class Mean_Teacher_dataset_1d_resample(Dataset):
    def __init__(self, feature_path, dev_path, label_path,
                 unlabeled_feature_path, unlabeled_dev_path):
        super().__init__()
        # train set
        X = np.load(feature_path + '_train.npy')
        dev = np.load(dev_path + '_train.npy')
        y = np.load(label_path + '_train.npy')

        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.train_set['x'] = X
        self.train_set['dev'] = dev
        self.train_set['y'] = y

        # unlabeled train set
        X_un = np.load(unlabeled_feature_path + '_train.npy')
        dev_un = np.load(unlabeled_dev_path + '_train.npy')
        assert X_un.shape[0] == dev_un.shape[0]

        self.unlabeled_train_set = np.zeros(X_un.shape[0], dtype=[
            ('x', np.float32, (X_un.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.unlabeled_train_set['x'] = X_un
        self.unlabeled_train_set['dev'] = dev_un
        self.unlabeled_train_set['y'] = -1

        # test_set
        X2 = np.load(feature_path + '_test.npy')
        dev2 = np.load(dev_path + '_test.npy')
        y2 = np.load(label_path + '_test.npy')
        assert X2.shape[0] == y2.shape[0]
        assert dev2.shape[0] == y2.shape[0]

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.test_set['x'] = X2
        self.test_set['dev'] = dev2
        self.test_set['y'] = y2

        self.test_set, self.val_set = train_test_split(self.test_set, test_size=10000, random_state=22)

    def labeled_pos_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.train_set) > 0
        anomaly = self.train_set[self.train_set['y'].flatten() == 1]
        for batch_idxs in random_index(len(anomaly), batch_size, random):
            yield anomaly[batch_idxs]

    def labeled_neg_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.train_set) > 0
        normal = self.train_set[self.train_set['y'].flatten() == 0]
        for batch_idxs in random_index(len(normal), batch_size, random):
            yield normal[batch_idxs]

    def unlabeled_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.unlabeled_train_set) > 0
        for batch_idxs in random_index(len(self.unlabeled_train_set), batch_size, random):
            yield self.unlabeled_train_set[batch_idxs]

    def training_generator(self, pos=50, neg=50, un=100):
        labeled_pos = self.labeled_pos_generator(batch_size=pos)
        labeled_neg = self.labeled_neg_generator(batch_size=neg)
        unlabeled = self.unlabeled_generator(batch_size=un)
        return zip(labeled_pos, labeled_neg, unlabeled)

class Mean_Teacher_dataset_2d(Dataset):
    def __init__(self, feature_path, dev_path, label_path,
                 unlabeled_feature_path, unlabeled_dev_path):
        super().__init__()
        # train set
        X = np.load(feature_path + '_train.npy')
        dev = np.load(dev_path + '_train.npy')
        y = np.load(label_path + '_train.npy')
        assert X.shape[0] == y.shape[0]
        assert dev.shape[0] == y.shape[0]
        self.train_set = np.zeros(X.shape[0], dtype=[
            ('x', np.float32, (X.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.train_set['x'] = X
        self.train_set['dev'] = dev
        self.train_set['y'] = y
        # # upsample 100 times
        # anomaly = self.train_set[self.train_set['y']==1]
        # anomaly = np.repeat(anomaly, 99)
        # self.train_set = np.concatenate([anomaly, self.train_set], axis=0)
        # np.random.shuffle(self.train_set)

        anomaly = self.train_set[self.train_set['y']==1]
        normal = self.train_set[self.train_set['y']==0]
        normal, _ = train_test_split(normal, train_size=anomaly.shape[0], random_state=12)
        self.balanced_train_set = np.concatenate([anomaly, normal], axis=0)

        # unlabeled train set
        X_un = np.load(unlabeled_feature_path + '_train.npy')
        dev_un = np.load(unlabeled_dev_path + '_train.npy')
        assert X_un.shape[0] == dev_un.shape[0]

        self.unlabeled_train_set = np.zeros(X_un.shape[0], dtype=[
            ('x', np.float32, (X_un.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.unlabeled_train_set['x'] = X_un
        self.unlabeled_train_set['dev'] = dev_un
        self.unlabeled_train_set['y'] = -1

        # test_set
        X2 = np.load(feature_path + '_test.npy')
        dev2 = np.load(dev_path + '_test.npy')
        y2 = np.load(label_path + '_test.npy')
        assert X2.shape[0] == y2.shape[0]
        assert dev2.shape[0] == y2.shape[0]

        self.test_set = np.zeros(X2.shape[0], dtype=[
            ('x', np.float32, (X2.shape[1:])),
            ('dev', np.int32, ()),
            ('y', np.int32, ())
        ])
        self.test_set['x'] = X2
        self.test_set['dev'] = dev2
        self.test_set['y'] = y2

        # self.test_set, self.val_set = train_test_split(self.test_set, test_size=10000, random_state=32)
        anomaly = self.test_set[self.test_set['y']==1]
        test_a, val_a = train_test_split(anomaly, test_size=20, random_state=5)
        normal = self.test_set[self.test_set['y']==0]
        test_n, val_n = train_test_split(normal, test_size=20, random_state=5)

        self.val_set = np.concatenate([val_a,val_n], axis=0)
        self.test_set = np.concatenate([test_a, test_n], axis=0)
        np.random.shuffle(self.test_set)

    def labeled_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.train_set) > 0
        for batch_idxs in random_index(len(self.train_set), batch_size, random):
            yield self.train_set[batch_idxs]

    def balance_labeled_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.balanced_train_set) > 0
        for batch_idxs in random_index(len(self.balanced_train_set), batch_size, random):
            yield self.balanced_train_set[batch_idxs]

    def unlabeled_generator(self, batch_size=50, random=np.random):
        assert batch_size > 0 and len(self.unlabeled_train_set) > 0
        for batch_idxs in random_index(len(self.unlabeled_train_set), batch_size, random):
            yield self.unlabeled_train_set[batch_idxs]

    def training_generator(self, batch_size=100, portion = 0.5):
        labeled = self.labeled_generator(batch_size=batch_size - int(batch_size*portion))
        unlabeled = self.unlabeled_generator(batch_size = int(batch_size*portion))
        return zip(labeled, unlabeled)

    def balance_training_generator(self, batch_size=100, portion = 0.5):
        labeled = self.balance_labeled_generator(batch_size=batch_size - int(batch_size*portion))
        unlabeled = self.unlabeled_generator(batch_size = int(batch_size*portion))
        return zip(labeled, unlabeled)
