import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
import os, random
from tqdm import tqdm 

import os
import os.path

import torch
from PIL import Image

from torchvision import datasets, transforms, models
import numpy as np
import abc
import pickle

from torch.utils import data as data_utils
from matplotlib import pyplot as plt
from mydatasets.skeleton import Skeleton

import pandas as pd

IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


"""
Adopted from Juyeon's contribution
"""
class CUBDataset(Dataset):
    def __init__(self, img_dir, img_list, image_transform, attr_group_dict=None, testing=False):
        with open(img_list, "rb" ) as f:
            self.labels = pickle.load(f)

        self.image_transform = image_transform
        self.img_dir = img_dir
        self.num_concepts = 112
        self.num_labels = 200

        # np.random.seed()
        self.attr_group_dict = attr_group_dict

        self.testing = testing
        self.epoch = 1
        self.class2concept= self._get_class2concept()
        self.concept_imb_ratio = self._get_concept_imbalance_ratio()

        self.bbox = self._load_bbox()


    def _get_class2concept(self):
        class2concept = torch.zeros(200, 112)
        for label in self.labels:
            class2concept[label['class_label']] = torch.Tensor(label['attribute_label'])
        class2concept[class2concept == 0] = -1
        return class2concept
    
    def _get_concept_imbalance_ratio(self):
        num_attr = torch.zeros(112)
        for label in self.labels:
            num_attr += torch.Tensor(label['attribute_label'])
        imbalance_ratio = len(self.labels) / num_attr - 1
        return imbalance_ratio

    def _load_bbox(self):
        root = self.img_dir[:self.img_dir.index('CUB_200_2011')]
        bbox = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'bounding_boxes.txt'), sep=' ', names=['img_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        bbox.img_id = bbox.img_id.astype(int)
        return bbox

    def __getitem__(self, index):
        name = self.labels[index]['img_path']
        if 'images' in name:
            name = name.replace('/juice/scr/scr102/scr/thaonguyen/CUB_supervision/mydatasets/CUB_200_2011/images/' ,'')
        img_path = os.path.join(self.img_dir, name)

        image = Image.open(img_path).convert('RGB')

        if self.image_transform is not None:
            image = self.image_transform(image)

        concept = torch.Tensor(self.labels[index]['attribute_label'])
        class_label = torch.Tensor([self.labels[index]['class_label']])
        concept_certainty = torch.Tensor(self.labels[index]['attribute_certainty'])

        sample = {}
        sample['image'] = image
        sample['concept_label'] = concept
        sample['class_label'] = class_label
        sample['concept_certainty'] = concept_certainty
        sample['imageID'] = name

        return sample

    def find_image(self, fname):
        for index, label in enumerate(self.labels):
            name = label['img_path']
            if 'images' in name:
                name = name.split('/')[-1]
            if name == fname:
                return index

    def get_image(self, fname):
        index = self.find_image(fname)
        return self.__getitem__(index)

    def __len__(self):
        return len(self.labels)


def get_data(args, is_return_group2concept=False):
    data_root = args.dataroot
    pkl_root = args.metadataroot
    batch_size = args.batch_size
    resol = args.img_size

    # This is zero-indexed (https://github.com/yewsiang/ConceptBottleneck/issues/15)
    attr2attrlabel = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91,
                      93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181,
                      183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253,
                      254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

    with open(data_root + 'CUB_200_2011/attributes/attributes.txt', 'r') as f:
        strings = f.readlines()

    attr_group_dict = {}
    attr_group_dict_name = {}
    for i, idx in enumerate(attr2attrlabel):
        label = strings[idx].split(' ')[-1].replace('\n', '')
        group = label.split('::')[0]
        if group in attr_group_dict.keys():
            attr_group_dict[group].append(i)
            attr_group_dict_name[group].append(label)
        else:
            attr_group_dict[group] = [i]
            attr_group_dict_name[group] = [label]
    
    workers = args.workers
    mean, std = IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

    group2concept = torch.zeros(len(attr_group_dict), len(attr2attrlabel))
    for g_idx, (g_name, g) in enumerate(attr_group_dict.items()):
        for c in g:
            group2concept[g_idx, c] = 1

    if is_return_group2concept:
        return group2concept

    if args.test_batch_size == -1:
        args.test_batch_size = batch_size

    train_dataset, val_dataset, test_dataset = None, None, None
    drop_last = True
    resized_resol = int(resol * 256 / 224)

#     trainTransform = transforms.Compose([
#         transforms.ColorJitter(brightness=32 / 255, saturation=0.5),
#         transforms.RandomHorizontalFlip(),
#         transforms.Resize((resized_resol, resized_resol)),
#         transforms.RandomResizedCrop(resol, scale=(0.8, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#         ])

#     testTransform = transforms.Compose([
#         transforms.Resize((resized_resol, resized_resol)),
#         transforms.CenterCrop(resol),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#         ])

    trainTransform, testTransform = transforms.Compose([transforms.Resize((resized_resol, resized_resol)), transforms.CenterCrop(resol), transforms.ToTensor()]), transforms.Compose([transforms.Resize((resized_resol, resized_resol)), transforms.CenterCrop(resol), transforms.ToTensor()])

    cub_root = os.path.join(data_root, 'CUB_200_2011')
    image_dir = os.path.join(cub_root, 'images')
    train_list = os.path.join(pkl_root, 'train.pkl')
    val_list = os.path.join(pkl_root, 'val.pkl')
    test_list = os.path.join(pkl_root, 'test.pkl')

    train_dataset = CUBDataset(image_dir, train_list, trainTransform, attr_group_dict=attr_group_dict, testing=False)

    image_dir = os.path.join(cub_root, 'images')
    val_dataset = CUBDataset(image_dir, val_list, testTransform, 
                                attr_group_dict=attr_group_dict, testing=True)
    test_dataset = CUBDataset(image_dir, test_list, testTransform, 
                                attr_group_dict=attr_group_dict, testing=True)
    return train_dataset, val_dataset, test_dataset, attr_group_dict_name


class _Wrapper(data_utils.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_classes = 200
        self.num_concepts = 112

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        concept = sample['concept_label']
        class_label = sample['class_label']
        concept_certainty = sample['concept_certainty']
        name = sample['imageID']
        
        return torch.Tensor(image), int(class_label), torch.Tensor(concept).type(torch.LongTensor)

    def __len__(self):
        return len(self.dataset)
        # return 10


class CUB(Skeleton):
    def __init__(self, fldr):
        class args():
            def __init__(self):
                self.dataroot = fldr
                self.metadataroot = f'{fldr}/CUB_200_2011/class_attr_data_10/'
                self.path_to_attributes=f'{fldr}/CUB_200_2011/attributes/'
                self.batch_size = 128
                self.test_batch_size = 128
                self.img_size = 224
                self.workers=4

        # has_upperparts_color::buff -> upperparts color is buff
        # has_size::medium_(9_-_16_in) -> size is medium (9 - 16 in)
        def clean_concept_name(cname):
            cname = cname.replace("has_", "")
            cname = cname.replace("_", " ")
            cname = cname.replace("::", " is ")
            cname = "image of a bird with " + cname
            return cname
            
        self.args = args()
        self.train_dataset, self.val_dataset, self.test_dataset, self.attr_group_dict_name = get_data(self.args)
        self.num_classes = 200
        self._concept_names = [clean_concept_name(cname) for cname in sum(self.attr_group_dict_name.values(), [])]
        self.num_concepts = 112

    def get_train_dataset(self):
        return _Wrapper(self.train_dataset)

    def get_val_dataset(self):
        return _Wrapper(self.val_dataset)
    
    def get_test_dataset(self):
        return _Wrapper(self.test_dataset)
    
    def num_classes(self):
        return 200
    
    @property
    def concept_names(self):
        return self._concept_names

    
def train_classifier(train_dat, val_dat, test_dat, model_save_name, num_classes=200):
    """
    Trains a ResNet-18 Model on the provided training dataset 
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    bmodel = models.__dict__['resnet18'](num_classes=num_classes)
    bmodel = bmodel.to(device)
    train_dl = data_utils.DataLoader(train_dat, batch_size=64, shuffle=True)
    val_dl = data_utils.DataLoader(val_dat, batch_size=64, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(bmodel.parameters(), lr=1e-3)
    num_epochs = 100
    for epoch in range(num_epochs):
        bmodel.train()
        for batch_x, batch_y, batch_g in tqdm(train_dl):
            logits = bmodel(batch_x.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        bmodel.eval()
        num_correct, num_total = 0, 0
        for batch_x, batch_y, batch_g in val_dl:
            preds = torch.argmax(bmodel(batch_x.to(device)), dim=-1).detach().cpu()
            num_correct += (preds == batch_y).sum()
            num_total += len(batch_x)
        print(f"Epoch: {epoch} Val. Acc.: {(num_correct/num_total)*100}")
    bmodel = bmodel.cpu()
    torch.save(bmodel.state_dict(), model_save_name)
    
    
if __name__ == '__main__':
    dat = CUB(fldr='/scratch/vp421/')
    img, y, g = dat.get_train_dataset()[0]
    print(len(dat.get_train_dataset()), img.shape, y, g.shape)
    print(len(dat.concept_names))

    train_classifier(dat.get_train_dataset(), dat.get_val_dataset(), dat.get_test_dataset(), 
                     num_classes=dat.num_classes, model_save_name='lightning_logs/resnet18_cub_2011_200.pt')
