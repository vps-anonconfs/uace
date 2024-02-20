import pickle
import torch

from mydatasets.broden_dataset import BrodenDataset
from explanations.concept_explainers import ConceptExplainer
from explanations.ocbm import cv_optimizer

import os
import pandas as pd
from clip import clip
from typing import List

from torchvision import transforms as T
from clip import clip
import torch
import torch.utils.data as data_utils
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        return torch.mm(clip_reprs, self.clip_embeddings.t())*100
        
        
def get_activations(context):
    rng = np.random.default_rng(context.random_state)
    _idxs = rng.permuted(np.arange(len(context.dataset)))
    _ln = int(len(_idxs) * 0.8)
    train_idxs, val_idxs = _idxs[:_ln], _idxs[_ln:]
    dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)

    # precompute P and reprs
    assert context.concepts and len(context.concepts) > 0, \
        f"Concepts param should be set to non-empty when using this explainer"
    clip_concepts = clip.tokenize(context.concepts).to(device)

    all_reprs, all_clip_sims, all_y, all_g = [], [], [], []
    _pfn = T.ToPILImage()
    for batch_x, batch_y, batch_g in tqdm(dl, desc="Computing P matrix"):
        with torch.no_grad():
            all_reprs.append(context.reprs_layer(batch_x).detach().cpu())
            logits_per_image = context.custom_clip_model(batch_x)
            
        _clip_sims = logits_per_image.detach().cpu() / 100
        all_clip_sims.append(_clip_sims)
        all_y.append(batch_y.cpu())
        all_g.append(batch_g.cpu())

    # all_reprs: N x repr_size, P: N x num_concepts
    all_reprs, P = torch.cat(all_reprs, dim=0), torch.cat(all_clip_sims, dim=0)
    mu, sigma = torch.mean(P, dim=0).unsqueeze(dim=0), torch.std(P, dim=0).unsqueeze(dim=0)
    P = (P - mu) / sigma

    all_y, all_g = torch.cat(all_y, dim=0), torch.cat(all_g, dim=0)
    train_reprs, train_P = all_reprs[train_idxs, :], P[train_idxs, :]
    val_reprs, val_P = all_reprs[val_idxs, :], P[val_idxs, :]

    repr_size, num_concepts = all_reprs.shape[-1], P.shape[-1]
    assert num_concepts == len(context.concepts)

    wc, cos_sims = cv_optimizer(context, train_reprs, train_P, val_reprs, val_P, use_cubic=False)

    # Get a normalized representation and make an indexed dataloader
    with torch.no_grad():
        X = torch.mm(all_reprs, wc).detach()
        y = all_y.detach()
        
    # return concept activations, clip similarities, concept-level cos(alpha_k)
    return X*sigma + mu, torch.cat(all_clip_sims, dim=0), cos_sims 


if __name__ == '__main__':
    """
    Evaluate confidence intervals obtained by U-ACE by comparison with the one obtained using MC sampling. 
    The intent is to empirically estimate the m(x), s(x) through MC sampling. 
    But that proved tricky because random sampling only led to more variance on relevant concepts that the model-to-be-explained actually understood.
    This may be because sampling mixes both data and epistemic uncertainty. 
    Since we estimate concept vectors through SGD, they have an implicit prior. 
    Averaging with a strong prior will only lead to consistent estimate for ill-defined concepts.
    
    Nevertheless, the empirical observation we made through MC sampling is conclusive that variance estimated using MC sampling is not capturing epistemic uncertainty and was infact inversely proportional to it. 
    """

    bfldr = "/scratch/vp421/data/broden1_224"
    cfname = f"{bfldr}/categories_places365.txt"
    if not os.path.exists(cfname):
        os.system(f"curl -o {cfname} https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt")
    cat365 = pd.read_csv(cfname, delimiter=' ').to_numpy()
    cat365_to_label = dict(zip(cat365[:, 0], cat365[:, 1]))
    cats_of_interest = ['/a/arena/hockey', '/a/auto_showroom', '/b/bedroom', '/c/conference_room', '/c/corn_field', '/h/hardware_store', '/l/legislative_chamber', '/t/tree_farm', '/c/coast']
    cats_of_interest += ['/p/parking_lot', '/p/pasture', '/p/patio', '/f/farm', '/p/playground', '/f/field/wild', '/p/playroom', '/f/forest_path', '/g/garage/indoor', '/g/garage/outdoor', '/r/runway', '/h/harbor', '/h/highway', '/b/beach', '/h/home_office', '/h/home_theater', '/s/slum', '/b/berth', '/s/stable', '/b/boat_deck', '/b/bow_window/indoor', '/s/street', '/s/subway_station/platform', '/b/bus_station/indoor', '/t/television_room', '/k/kennel/outdoor', '/c/campsite', '/l/lawn', '/t/tundra', '/l/living_room', '/l/loading_dock', '/m/marsh', '/w/waiting_room', '/c/computer_room', '/w/watering_hole', '/y/yard', '/n/nursery', '/o/office', '/d/dining_room', '/d/dorm_room', '/d/driveway']
    labels_of_interest = [cat365_to_label[_c] for _c in cats_of_interest]
 
    # "ade20k",
    sub_dataset = "pascal"
    splits = ["train", "val", "test"]
    cache_fldr = f"/scratch/vp421/data/{sub_dataset}"
    dat = BrodenDataset(sub_dataset, cache_fldr, bfldr)
    print(dat.concept_names)

    dummy_prediction_model = DummyModel([cache_fldr + f"/{split_name}_logits.pkl" for split_name in splits])
    dummy_representation_model = DummyModel([cache_fldr + f"/{split_name}_features.pkl" for split_name in splits])
    dummy_clip_model = DummyCLIPModel([cache_fldr + f"/{split_name}_clip.pkl" for split_name in splits], dat.concept_names)

    estimates = {}
    # with open("lightning_logs/eval_uncert.pkl", "rb") as f:
    #     estimates = pickle.load(f)
    for num_samples in [5, 10]:
        all_activations = []
        for sample_num in range(num_samples):
            print("-----------------------")
            print(f"Progress: {sample_num}/{num_samples}\n----------------")
            cb_explainer = ConceptExplainer(dummy_prediction_model, dummy_representation_model, None, 
                                            dat.get_train_dataset(), num_classes=dat.num_classes, 
                                            concepts=dat.concept_names, random_state=sample_num, 
                                            custom_clip_model=dummy_clip_model, labels_of_interest=labels_of_interest)

            activations, clip_activations, cos_sims = get_activations(cb_explainer)
            all_activations.append(activations)

        _a = torch.stack(all_activations, dim=0)
        m_e_x = torch.median(_a, dim=0)[0]
        s_e_x = torch.quantile(torch.abs(_a - torch.unsqueeze(m_e_x, dim=0)), 0.9, dim=0)

        cos_sims = torch.unsqueeze(torch.tensor(cos_sims), dim=0)
        theta = torch.arccos(clip_activations)
        alpha = torch.arccos(cos_sims)
        m_x = clip_activations*cos_sims
        s_x = torch.sin(theta)*torch.sin(alpha)

        print(m_e_x[:10, 30:50], s_e_x[:10, 30:50])
        print(m_x[:10, 30:50], s_x[:10, 30:50])
        
        estimates[num_samples] = (m_e_x, s_e_x, m_x, s_x)
        with open("lightning_logs/eval_uncert2.pkl", "wb") as f:
            pickle.dump(estimates, f)
        
        