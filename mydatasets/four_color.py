import os.path

import torch
from PIL import Image

from mydatasets.skeleton import Skeleton
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, STL10
import numpy as np

from torch.utils import data as data_utils
from matplotlib import pyplot as plt
from matplotlib import colors
from mydatasets import simple_tag_dataset


class _Wrapper(data_utils.Dataset):
    def __init__(self, x, y, g):
        self.x, self.y, self.g = x, y, g
        self.dat_transform = transforms.ToTensor()

    def __getitem__(self, idx):
        return self.dat_transform(self.x[idx]), self.y[idx], torch.Tensor(self.g[idx])

    def __len__(self):
        return len(self.x)


def _get_examples(N, mode_choice_prob, size):
    train_x, train_y, train_g = [], [], []
    for _i in range(N):
        y = np.random.choice(2)
        train_y.append(y)

        img = Image.new("RGB", (size, size))
        img = np.array(img)
        img = np.clip(np.random.normal(0, 20, size=img.shape), 0, 255)
        mode = np.random.choice(2, p=[mode_choice_prob, 1-mode_choice_prob])
        if y == 0:
            if mode == 0:
                img[:, :, 0] += 200
            else:
                img[:, :, 1] += 200
        else:
            if mode == 0:
                img[:, :, 2] += 200
            else:
                img[:, :, :] += 200
        img = Image.fromarray(img.astype(np.uint8))
        train_x.append(img)
        train_g.append(np.array([(1-y)*(1-mode), (1-y)*mode, y*(1-mode), y*mode]))

    return train_x, train_y, train_g


class MultiColorDataset(data_utils.Dataset):
    def __init__(self, num_concepts=None):
        self.colors = ["red", "green", "blue", "white"] + ["apple", "apricot", "avocado", "banana", "blackberry", "blueberry", "cantaloupe", "cherry", "coconut", "cranberry", "cucumber", "currant", "date", "dragonfruit", "durian", "elderberry", "fig", "grape", "grapefruit", "guava", "honeydew", "kiwi", "lemon", "lime", "loquat", "lychee", "mandarin orange", "mango", "melon", "nectarine", "orange", "papaya", "passion fruit", "peach", "pear", "persimmon", "pineapple", "plum", "pomegranate", "pomelo", "prune", "quince", "raspberry", "rhubarb", "star fruit", "strawberry", "tangerine", "tomato", "watermelon"]
        # ["orange", "yellow", "black", "gray", "cyan", "magenta", "gold", "silver", "olive", "lime", "teal", "maroon"]
        # ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        # ["lion", "tiger", "giraffe", "zebra", "monkey", "bear", "wolf", "fox", "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "deer", "rabbit", "raccoon", "squirrel", "mouse", "rat", "snake", "crocodile", "alligator", "turtle", "tortoise", "lizard", "chameleon", "iguana", "komodo dragon", "frog", "toad", "turtle", "tortoise", "leopard", "cheetah", "jaguar", "hyena", "wildebeest", "gnu", "bison", "antelope", "gazelle", "gemsbok", "oryx", "warthog", "hippopotamus", "rhinoceros", "elephant seal", "polar bear", "penguin", "flamingo", "ostrich", "emu", "cassowary", "kiwi", "koala", "wombat", "platypus", "echidna", "elephant"]
        # ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        # ["orange", "yellow", "black", "gray", "cyan", "magenta", "gold", "silver", "olive", "lime", "teal", "maroon"]
        # ["apple", "apricot", "avocado", "banana", "blackberry", "blueberry", "cantaloupe", "cherry", "coconut", "cranberry", "cucumber", "currant", "date", "dragonfruit", "durian", "elderberry", "fig", "grape", "grapefruit", "guava", "honeydew", "kiwi", "lemon", "lime", "loquat", "lychee", "mandarin orange", "mango", "melon", "nectarine", "orange", "papaya", "passion fruit", "peach", "pear", "persimmon", "pineapple", "plum", "pomegranate", "pomelo", "prune", "quince", "raspberry", "rhubarb", "star fruit", "strawberry", "tangerine", "tomato", "watermelon"]
        self.N = 1000
        self.num_concepts = num_concepts if num_concepts else len(self.colors)
        self.dat_transform = transforms.ToTensor()
    
    def __len__(self):
        return self.N
    
    @property
    def concept_names(self):
        return self.colors[:self.num_concepts]
    
    def __getitem__(self, idx):
        ridx = np.random.choice(len(self.concept_names))
        ridx = ridx % 4
        rcolor = self.concept_names[ridx]
        
        sz = 256
        img = Image.new("RGB", (sz, sz))
        img = np.array(img)
        img = np.clip(np.random.normal(0, 20, size=img.shape), 0, 255)
        
        code = colors.to_rgba(rcolor)
        for idx in range(3):
            img[:, :, idx] += code[idx] * 200
        img = Image.fromarray(img.astype(np.uint8))
        # else:
        #     img[:, :, ] += 200
        #     img = Image.fromarray(img.astype(np.uint8))
        #     draw = ImageDraw.Draw(img)
        #     font = ImageFont.truetype('arial.ttf', size=20)
        #     draw.text((2, 2), f"{rcolor}", fill="white", font=font)

        # img = np.transpose(img.astype(np.float32), [2, 0, 1])
        # img = Image.fromarray(img.astype(np.uint8))
        _g = np.zeros([self.num_concepts])
        _g[ridx] = 1
        _y = ridx // 2
        return self.dat_transform(img), _y, _g

    
class MultiColorDataset3(data_utils.Dataset):
    def __init__(self, num_concepts=None):
        self.colors = ["red or blue", "blue or red", "green or blue", "blue or green"]
        self.N = 1000
        self.num_concepts = num_concepts if num_concepts else len(self.colors)
        self.dat_transform = transforms.ToTensor()
    
    def __len__(self):
        return self.N
    
    @property
    def concept_names(self):
        return self.colors[:self.num_concepts]
    
    def __getitem__(self, idx):
        ridx = np.random.choice(4)
        ridx = ridx % 4
        rcolor = ["red", "green", "blue", "white"][ridx]
        
        sz = 256
        img = Image.new("RGB", (sz, sz))
        img = np.array(img)
        img = np.clip(np.random.normal(0, 20, size=img.shape), 0, 255)
        
        code = colors.to_rgba(rcolor)
        for idx in range(3):
            img[:, :, idx] += code[idx] * 200
        img = Image.fromarray(img.astype(np.uint8))

        _g = np.zeros([self.num_concepts])
        _g[ridx] = 1
        _y = ridx // 2
        return self.dat_transform(img), _y, _g


class MultiColorDataset2(data_utils.Dataset):
    def __init__(self, num_concepts=None):
        # self.colors = ["red", "green", "blue", "white", "orange", "yellow", "brown", "teal", "lime", "olive", "silver", "gold", "coral", "turquoise"]
        self.colors = ["red", "green", "blue", "white"] + ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
        # ["apple", "apricot", "avocado", "banana", "blackberry", "blueberry", "cantaloupe", "cherry", "coconut", "cranberry", "cucumber", "currant", "date", "dragonfruit", "durian", "elderberry", "fig", "grape", "grapefruit", "guava", "honeydew", "kiwi", "lemon", "lime", "loquat", "lychee", "mandarin orange", "mango", "melon", "nectarine", "orange", "papaya", "passion fruit", "peach", "pear", "persimmon", "pineapple", "plum", "pomegranate", "pomelo", "prune", "quince", "raspberry", "rhubarb", "star fruit", "strawberry", "tangerine", "tomato", "watermelon"]

        self.N = 1000
        self.num_concepts = num_concepts if num_concepts else len(self.colors)
        self.dat_transform = transforms.ToTensor()
        
        fldr = os.path.expanduser("~/datasets")
        select_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # self.target_fn = lambda _: {0: 0, 1: 1, 2: 2, 3: 3}[_]
        _dataset = STL10(root=fldr, split='test')

        _targets = simple_tag_dataset.get_targets(_dataset)
        slct_idxs = simple_tag_dataset.select_idxs(_targets, select_labels)
        self.stl_dataset = data_utils.Subset(_dataset, slct_idxs)
        label_to_idx = {}
        for xi, (x, y) in enumerate(self.stl_dataset):
            if y not in label_to_idx:
                label_to_idx[y] = []
            idxs = label_to_idx[y]
            idxs.append(xi)
            label_to_idx[y] = idxs
        self.label_to_idx = label_to_idx
    
    def __len__(self):
        return self.N
    
    @property
    def concept_names(self):
        return self.colors[:self.num_concepts]
    
    def __getitem__(self, idx):
        ridx = np.random.choice(len(self.concept_names)) % 4
        rcolor = self.concept_names[ridx]
        sz = 256
        img = Image.new("RGB", (sz, sz))
        code = colors.to_rgba(rcolor)
        img.paste((int(code[0]*255), int(code[1]*255), int(code[2]*255)), (0, 0, sz//2, sz))

        stl_idx = np.random.choice(self.label_to_idx[np.random.choice([[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]][ridx // 2])])
        stl_img, _ = self.stl_dataset[stl_idx]
        stl_img = stl_img.resize((sz//2, sz))
        img.paste(stl_img, (sz//2, 0))

        # img = np.transpose(img.astype(np.float32), [2, 0, 1])
        # img = Image.fromarray(img.astype(np.uint8))
        _g = np.zeros([self.num_concepts])
        _g[ridx] = 1
        _y = ridx // 2
        # if ridx < 2:
        #     _y = 0
        # elif ridx < 4:
        #     _y = 1
        # else:
        #     _y = np.random.choice([0, 1])
        return self.dat_transform(img), _y, _g
        

class FColor(Skeleton):
    """
    Dataset with two classes of MNIST juxtaposed.
    class 0 = red or green
    class 1 = blue or white
    """

    def __init__(self, test_mode_mix=0.5, train_mode_mix=0.5, num_train=1000, num_test=100, random_state=42, size=28, with_nuisance_concepts=False):
        """
        mode 0 : red (label 0) or blue (label 1)
        mode 1 : green (label 0)  or white (label 1)
        :param test_mode_mix: Number between 0 and 1 that controls the number of test examples from mode 0 to mode 1
        """
        self.test_mode_mix = test_mode_mix
        num_concepts = 4
        rng = np.random.default_rng(random_state)

        self.train_x, self.train_y, self.train_g = _get_examples(N=num_train, mode_choice_prob=train_mode_mix, size=size)
        self.test_x, self.test_y, self.test_g = _get_examples(N=num_test, mode_choice_prob=test_mode_mix, size=size)
        
        self.nuisance_concepts = with_nuisance_concepts

    def get_train_dataset(self):
        return _Wrapper(self.train_x, self.train_y, self.train_g)

    def get_test_dataset(self):
        return _Wrapper(self.test_x, self.test_y, self.test_g)

    @property
    def num_classes(self):
        return 2

    @property
    def concept_names(self):
        if self.nuisance_concepts:
            return ["red", "green", "blue", "white", "orange", "yellow", "brown", "teal", "lime", "olive", "silver", "gold", "coral", "turquoise"]#[chr(ord('A') + i) for i in range(26)]
        else:
            return ["red", "green", "blue", "white"]


if __name__ == '__main__':
    dat = FColor(test_mode_mix=0)
    print(len(dat.get_test_dataset()))
    print(np.sum(dat.test_g, axis=0))
    print(np.sum(dat.train_g, axis=0))
    tdat = dat.get_train_dataset()
    print(tdat[0][0].shape, tdat[-20][1])

    idx = np.random.choice(len(tdat))
    plt.imshow(torch.permute(tdat[idx][0], [1, 2, 0]))
    # plt.savefig()
    plt.show()
    print(tdat[idx][1], tdat[idx][2])
    tdat = dat.get_test_dataset()
    y_arr = []
    for _i in range(len(tdat)):
        _, y, _ = tdat[_i]
        y_arr.append(y)
    print(np.unique(y_arr, return_counts=True))