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

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

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
        
        
def get_uncert(context):
    rng = np.random.default_rng(context.random_state)
    _idxs = rng.permuted(np.arange(len(context.dataset)))
    _ln = int(len(_idxs) * 0.8)
    train_idxs, val_idxs = _idxs[:_ln], _idxs[_ln:]
    dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)

    all_reprs, all_y, all_g = [], [], []
    _pfn = T.ToPILImage()
    for batch_x, batch_y, batch_g in tqdm(dl, desc="Computing P matrix"):
        with torch.no_grad():
            all_reprs.append(context.reprs_layer(batch_x).detach().cpu())
        all_y.append(batch_y.cpu())
        all_g.append(batch_g.cpu())

    # all_reprs: N x repr_size, P: N x num_concepts
    all_reprs = torch.cat(all_reprs, dim=0)

    all_y, all_g = torch.cat(all_y, dim=0), torch.cat(all_g, dim=0)

    repr_size = all_reprs.shape[-1]
    uncert = []
    for gi in tqdm(range(all_g.shape[-1]), desc="Concepts: "):
        y0_idxs = np.where(all_g[:, gi]==0)[0]
        y1_idxs = np.where(all_g[:, gi]==1)[0]
        if len(y1_idxs) > 25:
            ln1, ln2 = int(0.8*len(y0_idxs)), int(0.8*len(y1_idxs))
            # to avoid confounding from the empirical estimation and do avoid skew
            ln1, ln2 = 20, 20
            train_idxs = np.concatenate([y0_idxs[:ln1], y1_idxs[:ln2]])
            val_idxs = np.concatenate([y0_idxs[ln1:], y1_idxs[ln2:]])
            train_reprs, train_g = all_reprs[train_idxs, :], all_g[train_idxs, :]
            val_reprs, val_g = all_reprs[val_idxs, :], all_g[val_idxs, :]

            scaler = preprocessing.StandardScaler().fit(train_reprs)
            clf = LogisticRegression(random_state=0, max_iter=500).fit(scaler.transform(train_reprs), train_g[:, gi])
            s1 = clf.score(scaler.transform(all_reprs[y1_idxs[ln2:]]), all_g[y1_idxs[ln2:], gi])
            s2 = clf.score(scaler.transform(all_reprs[y0_idxs[ln1:]]), all_g[y0_idxs[ln1:], gi])
            print(dat.concept_names[gi], (s1+s2)/2)
            uncert.append((s1+s2)/2)
        else:
            uncert.append(-1)

    return uncert, all_g.numpy().mean(axis=0)


def fit_distr(context):
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

    all_y = torch.cat(all_y, dim=0)
    train_reprs, train_P = all_reprs[train_idxs, :], P[train_idxs, :]
    val_reprs, val_P = all_reprs[val_idxs, :], P[val_idxs, :]
    repr_size, num_concepts = all_reprs.shape[-1], all_g[0].shape[-1]
    mu_W = torch.nn.Parameter(torch.zeros([repr_size, num_concepts]).to(device))
    sigma_W = torch.nn.Parameter((torch.randn([repr_size, num_concepts])*1e-1).to(device))
    opt = torch.optim.Adam(lr=1e-3, params=[mu_W, sigma_W])
    all_reprs, P = all_reprs.to(device), P.to(device)
    print(mu_W.requires_grad, sigma_W.requires_grad)
    beta=1
    for _ in tqdm(range(2000)):
        opt.zero_grad()
        # N x num_concepts, N x num_concepts
        mu, log_sigma = torch.mm(all_reprs, mu_W), torch.mm(all_reprs, sigma_W)
        distr = torch.distributions.normal.Normal(mu, torch.nn.functional.softplus(log_sigma))
        lp = -distr.log_prob(P)
        distr2 = torch.distributions.normal.Normal(torch.zeros_like(mu), torch.ones_like(log_sigma))
        kl = torch.distributions.kl.kl_divergence(distr, distr2)
        # print(lp.shape, kl.shape)
        loss = (lp + beta*kl).mean()
        # print(torch.abs(torch.autograd.grad(loss, [mu_W])[0]).mean())
        loss.backward()
        if _ % 500 == 0:
            print(f"Step: {_} Loss: {loss.detach().cpu().numpy(): 0.2f}, {lp.mean().detach().cpu().numpy(): 0.2f}, {kl.mean().detach().cpu().numpy(): 0.2f}")
        opt.step()
    return torch.nn.functional.softplus(torch.mm(all_reprs, sigma_W)).detach().cpu()


if __name__ == '__main__':
    """
    Evaluate confidence intervals obtained by U-ACE by comparing with the prediction accuracy obtainable through logistic regression on representations. 
    This script does the following:
    (a) get_uncert(): fits a logistic regressor on representations for each concept (using ground-truth annotations) and returns a proxy for how much information about the concept is encoded in the reprs. These are then used as ground-truth for evaluation.
    (b) fit_distr(): fits a normal distribution on representations to maximize the log-prob of clip scores. This is based on ProbCBM paper, and serves as an alternate for estimating the confidence intervals on concept activations. 
    
    (b) although better than simple MC sampling is still found to be worse than U-ACE. This may be because (b) estimates the uncert using a dot-product and therefore the error in uncert is proportional to the norm of sigma_W. Whereas U-ACE estimates the same through sin(theta)sin(alpha), we may have higher tolerance on error in the case of U-ACE because alpha is the only estimated quantity. 
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

    dummy_prediction_model = DummyModel([cache_fldr + f"/{split_name}_logits.pkl" for split_name in splits])
    dummy_representation_model = DummyModel([cache_fldr + f"/{split_name}_features.pkl" for split_name in splits])
    dummy_clip_model = DummyCLIPModel([cache_fldr + f"/{split_name}_clip.pkl" for split_name in splits], dat.concept_names)

    cb_explainer = ConceptExplainer(dummy_prediction_model, dummy_representation_model, None, 
                                    dat.get_train_dataset(), num_classes=dat.num_classes, 
                                    concepts=dat.concept_names, random_state=0, 
                                    custom_clip_model=dummy_clip_model, labels_of_interest=labels_of_interest)

    uncert, freq = get_uncert(cb_explainer)

    idxs = np.argsort(uncert)
    print([dat.concept_names[idx] for idx in idxs[-20:]])
    with open("lightning_logs/eval_uncert2.pkl", "wb") as f:
        pickle.dump([uncert, freq], f)

    std_dev = fit_distr(cb_explainer)
    with open("lightning_logs/eval_uncert2_v2_beta=1.pkl", "wb") as f:
        pickle.dump(std_dev, f)