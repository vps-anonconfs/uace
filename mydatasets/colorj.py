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
    def __init__(self, x, y, g):
        self.x, self.y, self.g = x, y, g
        self.dat_transform = transforms.ToTensor()

    def __getitem__(self, idx):
        return self.dat_transform(self.x[idx]), self.y[idx], torch.Tensor(self.g[idx])

    def __len__(self):
        return len(self.x)


def _get_examples(N, switch_prob):
    train_x, train_y, train_g = [], [], []
    for _i in range(N):
        y = np.random.choice(2)
        train_y.append(y)

        img = Image.new("RGB", (56, 56))
        img = np.array(img)
        img = np.clip(np.random.normal(0, 20, size=img.shape), 0, 255)
        if y == 0:
            img[:, :28, 0] += 200
        else:
            img[:, :28, :] += 200
        ry = np.random.choice(2, p=[switch_prob, 1 - switch_prob])
        if y == 0:
            if ry == 0:
                img[:, 28:, 1] += 200
            else:
                img[:, 28:, 2] += 200
        else:
            if ry == 1:
                img[:, 28:, 1] += 200
            else:
                img[:, 28:, 2] += 200
        img = Image.fromarray(img.astype(np.uint8))
        train_x.append(img)
        train_g.append(np.array([1 - y, (1-y) * (1 - ry) + y * ry, (1-y) * ry + y * (1 - ry), y]))

    return train_x, train_y, train_g


class ColorJ(Skeleton):
    """
    Dataset with two classes where each example contains two colors juxtaposed.
    class 0 = red + black
    class 1 = green + blue
    """

    def __init__(self, train_rand_prob=1., test_rand_prob=0.5):
        """
        :param train_rand_prob: probability of right part of the image corresponding to the label in train
        :param test_rand_prob:              and in test
        """
        self.num_concepts = 4
        self.train_x, self.train_y, self.train_g = _get_examples(N=1000, switch_prob=train_rand_prob)
        self.test_x, self.test_y, self.test_g = _get_examples(N=100, switch_prob=test_rand_prob)

    def get_train_dataset(self):
        return _Wrapper(self.train_x, self.train_y, self.train_g)

    def get_test_dataset(self):
        return _Wrapper(self.test_x, self.test_y, self.test_g)

    @property
    def num_classes(self):
        return 2

    @property
    def concept_names(self):
        return ["red", "green", "blue", "white"]


if __name__ == '__main__':
    dat = ColorJ(test_rand_prob=0.5)
    print(len(dat.get_test_dataset()))
    print(np.sum(dat.test_g, axis=0))
    print(np.sum(dat.train_g, axis=0))
    tdat = dat.get_train_dataset()
    print(tdat[0][0].shape, tdat[-20][1])

    idx = np.random.choice(len(tdat))
    plt.imshow(torch.permute(tdat[idx][0], [1, 2, 0]))
    plt.show()

    concepts = dat.concept_names
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