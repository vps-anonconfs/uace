import os
from typing import Tuple, List

from PIL import ImageDraw, ImageFont
import torch
from torchvision.datasets import CIFAR10, STL10
from torchvision import transforms
from torch.utils import data as data_utils
import numpy as np
from matplotlib import pyplot as plt

from mydatasets.skeleton import Skeleton

rng = np.random.default_rng(42)


def tag_fn(pil_img, text):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype('arial.ttf', size=20)
    draw.text((2, 2), f"{text}", fill="red", font=font)
    return pil_img


class Wrapper(data_utils.Dataset):
    def __init__(self, dataset, original_dataset, target_transform, tag_prob, concepts: Tuple, add_tag, only_A, only_B):
        self.dataset = dataset
        self.original_dataset = original_dataset
        self.target_transform = target_transform
        self.tag_prob = tag_prob
        self.pos_concepts, self.neg_concepts, self.tag_concepts = concepts
        self.dat_transform = transforms.ToTensor()
        self.add_tag, self.only_A, self.only_B = add_tag, only_A, only_B

    def __getitem__(self, item):
        x, y = self.dataset[item]
        y = self.target_transform(y)
        g = torch.zeros([len(self.original_dataset.concept_names)])
        c_idx = y

        if rng.random() < self.tag_prob:
            c_idx = 1 - c_idx
        if (self.only_A or self.only_B):
            if self.only_A:
                if c_idx == 0:
                    x = tag_fn(x, self.tag_concepts[c_idx])
            elif self.only_B:
                if c_idx == 1:
                    x = tag_fn(x, self.tag_concepts[c_idx])
        elif self.add_tag:
            x = tag_fn(x, self.tag_concepts[c_idx])

        # TODO: Object annotations are sloppy. Not all images of car may have a visible tail light.
        #  Could this be a problem?
        if y == 0:
            g[: len(self.neg_concepts)] = 1.
        else:
            g[len(self.neg_concepts): len(self.neg_concepts + self.pos_concepts)] = 1.
        if c_idx == 0:
            g[len(self.neg_concepts + self.pos_concepts)] = 1
        else:
            g[len(self.neg_concepts + self.pos_concepts) + 1] = 1
        return self.dat_transform(x), y, g

    def __len__(self):
        return len(self.dataset)

def select_idxs(labels, select_labels):
    idxs = [np.where(labels == sl)[0] for sl in select_labels]
    return np.concatenate(idxs, axis=0)

def get_targets(dataset):
    if hasattr(dataset, "targets"):
        return dataset.targets
    return np.array([y for x, y in dataset])
        

class SimpleTag(Skeleton):
    """
    A simple dataset for sanity checking.
    Defines a binary classification task where one of the classes may contain a spurious tag
    This dataset is inspired from sanity check dataset of TCAV (Been Kim et.al.)
    """
    def __init__(self, tag_switch_prob:float = 0, data_frac=1000, random_state=0, add_tag=True, only_A=False, only_B=False):
        """
        :param tag_switch_prob: fraction of examples with spurious tag in negative class
        :param data_frac: number of training examples shall be restricted to this number if positive
        :param add_tag: set this to false to get a dataset without captions (a debug option)
        :param only_A, only_B: again debug options, when either of them are set to true, then only that tag is added irrespective of add_tag option
        """
        assert not(only_A and only_B)
        global rng
        rng = np.random.default_rng(random_state)

        fldr = os.path.expanduser("~/datasets")
        self.tag_switch_prob = tag_switch_prob
        select_labels = [0, 2]
        self.target_fn = lambda _: {0: 0, 2: 1}[_]
        self.train_dataset = STL10(root=fldr, split='train', download=True)
        self.test_dataset = STL10(root=fldr, split='test')

        train_targets, test_targets = get_targets(self.train_dataset), get_targets(self.test_dataset)
        slct_idxs1, slct_idxs2 = select_idxs(train_targets, select_labels), select_idxs(test_targets, select_labels)
        self.train_dataset = data_utils.Subset(self.train_dataset, slct_idxs1)
        self.test_dataset = data_utils.Subset(self.test_dataset, slct_idxs2)

        # subset train data
        if data_frac > 0:
            self.train_dataset = data_utils.Subset(self.train_dataset, rng.choice(len(self.train_dataset), data_frac))
        
        self.n_concepts = []
        self.add_tag = add_tag
        self.only_A, self.only_B = only_A, only_B

    def get_train_dataset(self):
        return Wrapper(self.train_dataset, self, self.target_fn, self.tag_switch_prob,
                       (self.pos_concept_names(), self.neg_concept_names(), self.tag_concept_names()), 
                       add_tag=self.add_tag, only_A=self.only_A, only_B=self.only_B)

    def get_test_dataset(self, tag_prob=None):
        if not tag_prob:
            tag_prob = self.tag_switch_prob
        return Wrapper(self.test_dataset, self, self.target_fn, tag_prob,
                       (self.pos_concept_names(), self.neg_concept_names(), self.tag_concept_names()), 
                       add_tag=self.add_tag, only_A=self.only_A, only_B=self.only_B)

    @property
    def num_classes(self):
        return 2

    @staticmethod
    def pos_concept_names():
        return ["headlights", "taillights", "turn signals", "windshield", "windshield vipers", "bumpers", "wheels"]

    @staticmethod
    def neg_concept_names():
        return ["wings", "landing gear", "sky"]

    @staticmethod
    def tag_concept_names():
        # U
        return ["U", "Z"]

    def set_nuisance_concept_names(self, n_concepts: List):
        self.n_concepts = n_concepts
        
    def nuisance_concept_names(self):
        return self.n_concepts
    
    @property
    def concept_names(self):
        concept_tokens = self.neg_concept_names()
        concept_tokens += self.pos_concept_names()
        concept_tokens += self.tag_concept_names()
        concept_tokens += self.nuisance_concept_names()
        return concept_tokens


if __name__ == '__main__':
    dat = SimpleTag(tag_switch_prob=0.5)
    _d = dat.get_test_dataset()
    idx = np.random.choice(len(_d))
    for _ in range(np.random.choice(len(_d))):
        _d[_]
    plt.imshow(torch.permute(_d[idx][0], [1, 2, 0]))
    plt.show()

    all_g, all_y = [], []
    for idx in range(len(_d)):
        all_g.append(_d[idx][-1])
        all_y.append(_d[idx][-2])
    all_y = np.array(all_y)
    all_g = torch.stack(all_g, dim=0).numpy()
    print(np.corrcoef(all_g[:, -1], all_y)[1, 0])
    print(np.corrcoef(all_g[:, -2], all_y)[1, 0])