import pickle

import torch
import torchvision

from mydatasets.broden_dataset import BrodenDataset
from explanations.concept_explainers import ConceptExplainer, NOISE_MODELLING

import os
import pandas as pd
from clip import clip
from typing import List
import numpy as np


class DummyModel:
    def __init__(self, cache_fnames: List):
        self.cache = {}
        for cache_fname in cache_fnames:
            with open(cache_fname, "rb") as f:
                x = pickle.load(f)
                self.cache = {**self.cache, **x}

    def __call__(self, fnames):
        return torch.stack([torch.tensor(self.cache[fname]) for fname in fnames], dim=0)


# normalize on second dimension
def _norm(mat):
    mat /= torch.linalg.norm(mat, dim=-1).unsqueeze(dim=-1)
    return mat


class DummyCLIPModel:
    def __init__(self, cache_fnames: List, concepts: List[str]):
        self.cache = {}
        for cache_fname in cache_fnames:
            with open(cache_fname, "rb") as f:
                x = pickle.load(f)
                self.cache = {**self.cache, **x}

        self.concepts = concepts
        self.clip_concepts = clip.tokenize(self.concepts)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device='cpu')
        with torch.no_grad():
            # num_concepts x clip_embedding_size
            self.clip_embeddings = clip_model.encode_text(self.clip_concepts)
            self.clip_embeddings = _norm(self.clip_embeddings)

    def __call__(self, fnames):
        # batch_size x clip_embedding_size
        clip_reprs = torch.stack([torch.tensor(self.cache[fname]) for fname in fnames], dim=0)
        # without type casting, clip_reprs are float16 and that is throwing an error
        clip_reprs = _norm(clip_reprs).type(torch.float32)

        # batch_size x num_concepts
        # multiplying by 100 just like clip does: https://github.com/openai/CLIP/tree/main#modelimage-tensor-text-tensor
        return torch.mm(clip_reprs, self.clip_embeddings.t()) * 100


if __name__ == '__main__':
    """
    Computes explanations for Places365-ResNet18 using ADE and PASCAL as probe mydatasets 
    """

    bfldr = "/scratch/vp421/data/broden1_224"
    cfname = f"{bfldr}/categories_places365.txt"
    if not os.path.exists(cfname):
        os.system(
            f"curl -o {cfname} https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt")
    cat365 = pd.read_csv(cfname, delimiter=' ').to_numpy()
    cat365_to_label = dict(zip(cat365[:, 0], cat365[:, 1]))
    cats_of_interest = ['/a/arena/hockey', '/a/auto_showroom', '/b/bedroom', '/c/conference_room', '/c/corn_field',
                        '/h/hardware_store', '/l/legislative_chamber', '/t/tree_farm', '/c/coast']
    # cats_of_interest += ['/c/coast', "/b/banquet_hall", "/h/hospital", "/m/mountain_snowy", "/r/restaurant", "/w/wheat_field", "/s/server_room", "/j/jewelry_shop", "/b/boathouse", "/a/art_studio"]
    # labels with at least 20 examples in both ade20k and pascal
    cats_of_interest += ['/p/parking_lot', '/p/pasture', '/p/patio', '/f/farm', '/p/playground', '/f/field/wild',
                         '/p/playroom', '/f/forest_path', '/g/garage/indoor', '/g/garage/outdoor', '/r/runway',
                         '/h/harbor', '/h/highway', '/b/beach', '/h/home_office', '/h/home_theater', '/s/slum',
                         '/b/berth', '/s/stable', '/b/boat_deck', '/b/bow_window/indoor', '/s/street',
                         '/s/subway_station/platform', '/b/bus_station/indoor', '/t/television_room',
                         '/k/kennel/outdoor', '/c/campsite', '/l/lawn', '/t/tundra', '/l/living_room',
                         '/l/loading_dock', '/m/marsh', '/w/waiting_room', '/c/computer_room', '/w/watering_hole',
                         '/y/yard', '/n/nursery', '/o/office', '/d/dining_room', '/d/dorm_room', '/d/driveway']
    labels_of_interest = [cat365_to_label[_c] for _c in cats_of_interest]

    # "ade20k",
    for sub_dataset in ["pascal", "ade20k"]:
        splits = ["train", "val", "test"]
        cache_fldr = f"/scratch/vp421/data/{sub_dataset}"
        dat = BrodenDataset(sub_dataset, cache_fldr, bfldr, for_cache=False)
        print(dat.concept_names)

        dummy_prediction_model = DummyModel([cache_fldr + f"/{split_name}_logits.pkl" for split_name in splits])
        dummy_representation_model = DummyModel([cache_fldr + f"/{split_name}_features.pkl" for split_name in splits])
        dummy_clip_model = DummyCLIPModel([cache_fldr + f"/{split_name}_clip.pkl" for split_name in splits],
                                          dat.concept_names)

        fldr = '/scratch/vp421/data'
        arch = 'resnet18'
        model_file = f'{fldr}/{arch}_places365.pth.tar' 
        places_model = torchvision.models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        places_model.load_state_dict(state_dict)
        
        random_state = 0
        cbe = {}
        cb_explainer = ConceptExplainer(places_model, dummy_representation_model, None,
                                        dat.get_train_dataset(), num_classes=dat.num_classes,
                                        concepts=dat.concept_names, random_state=random_state,
                                        custom_clip_model=dummy_clip_model, labels_of_interest=labels_of_interest)

        # layer4 is the last layer
        wts = cb_explainer.tcav(layer_index=-3)
        for ci in range(len(cats_of_interest)):
            print("--------------")
            print("Label: ", cats_of_interest[ci])
            sidxs = np.argsort(-wts[ci])
            for sidx in sidxs[:10]:
                print(dat.concept_names[sidx], wts[ci, sidx])
        
        cbe['tcav'] = wts
        model_dir = "lightning_logs"
        with open(f"{model_dir}/expte_{sub_dataset}_tcav.pkl", "wb") as f:
            pickle.dump([cbe], f)