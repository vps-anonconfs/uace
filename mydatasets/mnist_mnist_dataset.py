import os.path

import torch

from mydatasets.skeleton import Skeleton
from torchvision import datasets, transforms
import numpy as np

from torch.utils import data as data_utils
from matplotlib import pyplot as plt


class _Wrapper(data_utils.Dataset):
    def __init__(self, base_dataset, augment_arr):
        self.base_dat = base_dataset
        self.new_info = augment_arr
        self.dat_transform = transforms.ToTensor()
        self.label_transform = lambda _: _//2

    def __getitem__(self, idx):
        x, y = self.base_dat[idx]
        return self.dat_transform(x), self.label_transform(y), torch.Tensor(self.new_info[idx])

    def __len__(self):
        return len(self.base_dat)


def _get_idxs(labels):
    idxs0, idxs1 = np.where(labels == 0)[0], np.where(labels == 1)[0]
    idxs2, idxs3 = np.where(labels == 2)[0], np.where(labels == 3)[0]
    return idxs0, idxs1, idxs2, idxs3


class MNIST_MNIST(Skeleton):
    """
    Dataset with two classes of MNIST juxtaposed.
    class 0 = MNIST 0 or MNIST 1
    class 1 = MNIST 2 or MNIST 3
    """

    def __init__(self, test_mode_mix=0.5, num_train=5000, num_test=1000, random_state=42):
        """
        mode 0 : mnist 0 + empty (label 0) or mnist 2 + empty (label 1)
        mode 1 : empty + mnist 1 (label 0)  or empty + mnist 3 (label 1)
        :param test_mode_mix: Number between 0 and 1 that controls the number of test examples from mode 0 to mode 1
        """
        self.test_mode_mix = test_mode_mix
        self.mnist_dataset = datasets.MNIST(root = os.path.expanduser("~/datasets"))
        num_concepts = 4
        rng = np.random.default_rng(random_state)

        train_0, train_1, train_2, train_3 = _get_idxs(self.mnist_dataset.train_labels)
        test_0, test_1, test_2, test_3 = _get_idxs(self.mnist_dataset.test_labels)
        _t0, _t1 = rng.choice(train_0, num_train//2), rng.choice(train_1, num_train//2)
        _t2, _t3 = rng.choice(train_2, num_train // 2), rng.choice(train_3, num_train // 2)
        train_mix = 0.5
        train_mode1_ln = int(num_train * train_mix / 2)
        label1_idxs = np.concatenate([_t0[:train_mode1_ln], _t1[train_mode1_ln:]])
        label2_idxs = np.concatenate([_t2[:train_mode1_ln], _t3[train_mode1_ln:]])
        self.train_idxs = np.concatenate([label1_idxs, label2_idxs])
        self.train_gs = np.zeros([len(self.train_idxs), num_concepts])
        self.train_gs[np.arange(train_mode1_ln), 0] = 1
        self.train_gs[np.arange(train_mode1_ln, len(label1_idxs)), 1] = 1
        _l1 = len(label1_idxs)
        self.train_gs[np.arange(_l1, _l1 + train_mode1_ln), 2] = 1
        self.train_gs[np.arange(_l1 + train_mode1_ln, _l1 + len(label2_idxs)), 3] = 1

        _t0, _t1 = rng.choice(test_0, num_test // 2), rng.choice(test_1, num_test // 2)
        _t2, _t3 = rng.choice(test_2, num_test // 2), rng.choice(test_3, num_test // 2)
        test_mode1_ln = int(num_test * self.test_mode_mix / 2)
        label1_idxs = np.concatenate([_t0[:test_mode1_ln], _t1[test_mode1_ln:]])
        label2_idxs = np.concatenate([_t2[:test_mode1_ln], _t3[test_mode1_ln:]])
        self.test_idxs = np.concatenate([label1_idxs, label2_idxs])

        self.test_gs = np.zeros([len(self.test_idxs), num_concepts])
        self.test_gs[np.arange(test_mode1_ln), 0] = 1
        self.test_gs[np.arange(test_mode1_ln, len(label1_idxs)), 1] = 1
        _l1 = len(label1_idxs)
        self.test_gs[np.arange(_l1, _l1 + test_mode1_ln), 2] = 1
        self.test_gs[np.arange(_l1 + test_mode1_ln, _l1 + len(label2_idxs)), 3] = 1

    def get_train_dataset(self):
        return _Wrapper(data_utils.Subset(self.mnist_dataset, self.train_idxs), self.train_gs)

    def get_test_dataset(self):
        return _Wrapper(data_utils.Subset(self.mnist_dataset, self.test_idxs), self.test_gs)

    @property
    def num_classes(self):
        return 2


if __name__ == '__main__':
    dat = MNIST_MNIST(test_mode_mix=0.1)
    print(len(dat.get_test_dataset()))
    print(np.sum(dat.test_gs, axis=0))
    print(np.sum(dat.train_gs, axis=0))
    tdat = dat.get_train_dataset()
    print(tdat[0][0].shape, tdat[-20][1])

    idx = np.random.choice(len(tdat))
    plt.imshow(torch.permute(tdat[idx][0], [1, 2, 0]))
    plt.show()
    print(tdat[idx][1], tdat[idx][2])
    tdat = dat.get_test_dataset()
    y_arr = []
    for _i in range(len(tdat)):
        _, y, _ = tdat[_i]
        y_arr.append(y)
    print(np.unique(y_arr, return_counts=True))