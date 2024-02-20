import os.path

import torch
import torchvision.transforms
from PIL import Image
from clip import clip

from mydatasets.skeleton import Skeleton
from torchvision import datasets, transforms
import numpy as np

from torch.utils import data as data_utils
from matplotlib import pyplot as plt


class _Wrapper(data_utils.Dataset):
    def __init__(self, base_dataset, left_idxs, right_idxs, augment_arr):
        self.base_dat = base_dataset
        self.new_info = augment_arr
        self.left_idxs, self.right_idxs = left_idxs, right_idxs
        self.dat_transform = transforms.ToTensor()
        self.label_transform = lambda _: _//2

    def __getitem__(self, idx):
        xl, yl = self.base_dat[self.left_idxs[idx]]
        xr, yr = self.base_dat[self.right_idxs[idx]]
        xl = xl.resize((28, 56))
        xr = xr.resize((28, 56))
        new_img = Image.new('L', (56, 56))
        new_img.paste(xl, (0, 0))
        new_img.paste(xr, (28, 0))

        return self.dat_transform(new_img), self.label_transform(yl), torch.Tensor(self.new_info[idx])

    def __len__(self):
        return len(self.left_idxs)


def _get_idxs(labels):
    idxs0, idxs1 = np.where(labels == 0)[0], np.where(labels == 1)[0]
    idxs2, idxs3 = np.where(labels == 2)[0], np.where(labels == 3)[0]
    return idxs0, idxs1, idxs2, idxs3


class MNISTJ(Skeleton):
    """
    Dataset with two classes of MNIST juxtaposed.
    class 0 = MNIST 0 + MNIST 1
    class 1 = MNIST 2 + MNIST 3
    """

    def __init__(self, train_rand_prob=1., test_rand_prob=0.5, num_train=5000, num_test=1000, random_state=42):
        """
        MNIST 0, 1 (juxtaposed) with MNIST 2, 3 dataset with two classes.
        right part is randomized with probability provided in args
        :param train_rand_prob: probability of mnist part corresponding to the label in train
        :param test_rand_prob:              and in test
        :param num_train:
        :param num_test:
        """
        self.mnist_dataset = datasets.MNIST(root = os.path.expanduser("~/datasets"))
        rng = np.random.default_rng(random_state)
        train_0, train_1, train_2, train_3 = _get_idxs(self.mnist_dataset.train_labels)
        test_0, test_1, test_2, test_3 = _get_idxs(self.mnist_dataset.test_labels)
        num_concepts = 4

        self.train_left = np.concatenate([rng.choice(train_0, num_train//2), rng.choice(train_2, num_train//2)])
        _t0, _t1 = rng.choice(train_1, num_train), rng.choice(train_3, num_train)
        idxs0 = rng.choice(2, p=[train_rand_prob, 1-train_rand_prob], size=num_train//2)
        idxs1 = rng.choice(2, p=[1-train_rand_prob, train_rand_prob], size=num_train // 2)
        self.train_right = np.stack([_t0, _t1], axis=-1)[np.arange(num_train), np.concatenate([idxs0, idxs1])]
        self.train_gs = np.zeros([num_train, num_concepts])
        self.train_gs[np.arange(num_train // 2), 0] = 1
        self.train_gs[np.arange(num_train // 2, num_train), 2] = 1
        _idxs = 1 + 2*np.concatenate([idxs0, idxs1])
        self.train_gs[np.arange(num_train), _idxs] = 1

        self.test_left = np.concatenate([rng.choice(test_0, num_test//2), rng.choice(test_2, num_test//2)])
        _t0, _t1 = rng.choice(test_1, num_test), rng.choice(test_3, num_test)
        idxs0 = rng.choice(2, p=[test_rand_prob, 1-test_rand_prob], size=num_test // 2)
        idxs1 = rng.choice(2, p=[1-test_rand_prob, test_rand_prob], size=num_test // 2)
        self.test_right = np.stack([_t0, _t1], axis=-1)[np.arange(num_test), np.concatenate([idxs0, idxs1])]
        self.test_gs = np.zeros([num_test, num_concepts])
        self.test_gs[np.arange(num_test // 2), 0] = 1
        self.test_gs[np.arange(num_test // 2, num_test), 2] = 1
        self.test_gs[np.arange(num_test), 1 + 2 * np.concatenate([idxs0, idxs1])] = 1

    def get_train_dataset(self):
        return _Wrapper(self.mnist_dataset, self.train_left, self.train_right, self.train_gs)

    def get_test_dataset(self):
        return _Wrapper(self.mnist_dataset, self.test_left, self.test_right, self.test_gs)

    @property
    def num_classes(self):
        return 2

    @property
    def concept_names(self):
        return ["zero", "one", "two", "three"]


if __name__ == '__main__':
    dat = MNISTJ(test_rand_prob=0.5)
    print(len(dat.get_test_dataset()))
    print(np.sum(dat.test_gs, axis=0))
    print(np.sum(dat.train_gs, axis=0))
    tdat = dat.get_train_dataset()
    print(tdat[0][0].shape, tdat[-20][1])

    idx = np.random.choice(len(tdat))
    plt.imshow(torch.permute(tdat[idx][0], [1, 2, 0]))
    plt.show()

    concepts = ["zero", "one", "two", "three"]
    clip_model, clip_preprocess = clip.load("ViT-B/32", device='cpu')
    clip_concepts = clip.tokenize(concepts)
    _pfn = torchvision.transforms.ToPILImage()
    clip_batch = clip_preprocess(_pfn(tdat[idx][0])).unsqueeze(dim=0)
    logits, _ = clip_model(clip_batch, clip_concepts)
    print(logits)

    print(tdat[idx][1], tdat[idx][2])
    tdat = dat.get_test_dataset()
    y_arr = []
    for _i in range(len(tdat)):
        _, y, _ = tdat[_i]
        y_arr.append(y)
    print(np.unique(y_arr, return_counts=True))

    arr = []
    for _i in range(len(tdat)):
        x, y, g = tdat[_i]
        arr.append(np.concatenate([g[:2] == 1-y, g[2:] == y]))

    print(np.array(arr).mean(axis=0))