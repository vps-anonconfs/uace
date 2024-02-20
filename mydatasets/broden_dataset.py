import torch

import numpy as np
import pickle
import pandas as pd
import regex
from PIL import Image
from tqdm import tqdm

from torch.utils import data as data_utils

from mydatasets.skeleton import Skeleton
from mydatasets.broden_features import center_crop


def _get_examples(dataset_name, cache_fldr):
    with open(f"{cache_fldr}/{dataset_name}_imagelabels.pkl", "rb") as f:
        x = pickle.load(f)
        # pd series to dict. image name -> attr list
        all_attrs = x['labels'].to_dict()
        train_instances, val_instances, test_instances = [x[split] for split in ["train", "val", "test"]]
        
    # dictionary that goes from image name to scene label
    all_labels = {}
    for split_name in ["train", "val", "test"]:
        with open(f"{cache_fldr}/{split_name}_scene.pkl", "rb") as f:
            x = pickle.load(f)
            all_labels = dict(**all_labels, **x)
        
    return train_instances, val_instances, test_instances, all_labels, all_attrs


class _Wrapper(data_utils.Dataset):
    def __init__(self, fnames, y, g, num_concepts, idx_mapping, for_cache=True):
        self.g, self.fnames = g, fnames
        self.y = y
        self.num_concepts = num_concepts
        self.idx_mapping = idx_mapping
        self.for_cache = for_cache
        
    def __getitem__(self, idx):
        # dummy content label
        attrs = self.g[self.fnames[idx]]
        this_g = torch.zeros([self.num_concepts])
        for attr in attrs:
            # the concept is removed otherwise
            if self.idx_mapping:
                this_g[self.idx_mapping[attr]] = 1
        if not self.for_cache:
            fname = self.fnames[idx]
            _x = center_crop(Image.open(fname).convert('RGB'))
        else:
            _x = self.fnames[idx]
        return _x, self.y[self.fnames[idx]], this_g

    def __len__(self):
        return len(self.fnames)


class BrodenDataset(Skeleton):
    def __init__(self, sub_dataset_name, cache_fldr, broden_fldr, labels_of_interest=None, random_state=42, for_cache=True):
        """
        :param sub_dataset_name: only ade20k and pascal supported
        :param cache_fldr: Folder containing all cache related to sub_dataset_name
        :param broden_fldr: is needed just to read concept names
        :param labels_of_interest: scene labels of interest
        :param random_state: no randomness involved
        :param for_cache: if set to true x is set to fnames else set to pixel images
        """
        rng = np.random.default_rng(random_state)

        self.train_fnames, self.val_fnames, self.test_fnames, self.scene_labels, self.attrs = _get_examples(sub_dataset_name, cache_fldr)
        labels = pd.read_csv(f"{broden_fldr}/label.csv").to_numpy()
        self._concept_names = list(labels[:, 1])
        # because broden labels start from 1
        self._concept_names = ['None'] + self._concept_names
        
        # if labels_of_interest:
        #     found_attrs = []
        #     for k, scene_label in self.scene_labels.items():
        #         if scene_label in labels_of_interest:
        #             found_attrs += self.attrs[k]
        #     attrs_of_interest = np.unique(found_attrs)
        #     for _idx in range(len(self._concept_names)):
        #         print(attrs_of_interest, _idx)
        #         if _idx not in attrs_of_interest:
        #             self._concept_names[_idx] = 'None'
        #     print(f"Setting {len(attrs_of_interest)} attrs of interest")
                
        # original concept names have '-c' or '-s' etc. to identify different colors, scenes etc., which may throw CLIP off, so, redacting them
        patt = regex.compile("-[a-z]$")
        # self._concept_names = [patt.sub('', _c) for _c in self._concept_names]
        patt = regex.compile("-c$")
        self._concept_names = [patt.sub('', _c) for _c in self._concept_names]
        # because we are using concepts to explain scene labels, having scene concepts is pointless.
        filtered_concepts, self.old_to_new_idx = [], []
        for ci in range(len(self._concept_names)):
            new_idx = None
            if not self._concept_names[ci].endswith('-s'):
                new_idx = len(filtered_concepts)
                filtered_concepts.append(self._concept_names[ci])
            self.old_to_new_idx.append(new_idx)
        self._concept_names = filtered_concepts
        self.num_concepts = len(self.concept_names)

        self.for_cache = for_cache
                
    def get_train_dataset(self):
        return _Wrapper(self.train_fnames, self.scene_labels, self.attrs, self.num_concepts, self.old_to_new_idx, self.for_cache)

    def get_val_dataset(self):
        return _Wrapper(self.val_fnames, self.scene_labels, self.attrs, self.num_concepts, self.old_to_new_idx, self.for_cache)    
    
    def get_test_dataset(self):
        return _Wrapper(self.test_fnames, self.scene_labels, self.attrs, self.num_concepts, self.old_to_new_idx, self.for_cache)
    
    @property
    def concept_names(self):
        return self._concept_names


if __name__ == '__main__':
    dat = BrodenDataset('ade20k', cache_fldr='/scratch/vp421/data/ade20k', broden_fldr='/scratch/vp421/data/broden1_224', for_cache=False)
    
    train_data = dat.get_train_dataset()
    # val_data = dat.get_val_dataset()
    # test_data = dat.get_test_dataset()
    x, y, g = train_data[0]
    print(x.shape, g.shape)