import os, time
from PIL import Image
import tqdm
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from torchvision.models import ResNet18_Weights
from torchvision.transforms import transforms

from mydatasets.salient_imagenet import SalientImagenet, CLASS_NAMES
from explanations.concept_explainers import ConceptExplainer
from explanations.uace import NOISE_MODELLING

from clip import clip
from typing import List


def create_cache(model_name, cache_fldr, device='cuda'):
    from torchvision.models import feature_extraction as fe
    os.makedirs(cache_fldr, exist_ok=True)

    mex_model = models.get_model(model_name, pretrained=True)
    mex_model = fe.create_feature_extractor(mex_model, {'avgpool': 'out1', 'fc': 'out2'})
    mex_model = mex_model.to(device)
    mex_model.eval()

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    snet = SalientImagenet(return_image=True, return_fname=True)
    snet_dat = snet.get_test_dataset()
    snet_dl = DataLoader(snet_dat, batch_size=32)
    for name, m in [('clip', clip_model), ('features', mex_model)]:
        with torch.no_grad():
            m = m.to(device)

            count = 0
            start_time = time.time()
            img_to_logits, img_to_feature = {}, {}

            for batch_imgs, batch_img_names, batch_labels, dummy in tqdm.tqdm(snet_dl, desc=f"{name}"):
                if name == 'clip':
                    clip_batch = torch.stack([clip_preprocess(Image.open(img_path).convert('RGB')) for img_path in batch_img_names], dim=0).to(device)

                    clip_features = clip_model.encode_image(clip_batch)
                    clip_features = clip_features.detach().cpu().numpy()
                    for ii, img_path in enumerate(batch_img_names):
                        img_to_feature[img_path] = clip_features[ii].squeeze()

                else:
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    res = m(normalize(batch_imgs).to(device))
                    features = res['out1'].detach().cpu().numpy()
                    logits = res['out2'].detach().cpu().numpy()
                    for ii, img_path in enumerate(batch_img_names):
                        img_to_feature[img_path] = features[ii].squeeze()
                        img_to_logits[img_path] = np.array([logits[ii][cidx] for cidx in snet.class_idxs])

                count += 1
                if count % 1000 == 0:
                    elapsed_time = (time.time() - start_time) / 60
                    print('Processed {} images in {:.2f} minutes'.format(count, elapsed_time), flush=True)

            if name == 'clip':
                with open(f'{cache_fldr}/' + 'clip_features.pkl', 'wb+') as f:
                    pickle.dump(img_to_feature, f)
            else:
                with open(f'{cache_fldr}/' + 'mex_features.pkl', 'wb+') as f:
                    pickle.dump(img_to_feature, f)
                with open(f'{cache_fldr}/' + 'mex_logits.pkl', 'wb+') as f:
                    pickle.dump(img_to_logits, f)


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
    Computes explanations for images from Salient-Imagenet
    """
    cache_fldr = "/scratch/vp421/data/simagenet"
    dat = SalientImagenet(return_fname=True, return_image=False, return_mask=False)
    model_name = "resnet18"
    models.get_model(model_name, weights=ResNet18_Weights.DEFAULT)
    cache_fldr = os.path.join(cache_fldr, model_name)
    if not os.path.exists(cache_fldr):
        create_cache(model_name, cache_fldr)

    dummy_prediction_model = DummyModel([os.path.join(cache_fldr, "mex_logits.pkl")])
    dummy_representation_model = DummyModel([os.path.join(cache_fldr, "mex_features.pkl")])
    dummy_clip_model = DummyCLIPModel([os.path.join(cache_fldr, "clip_features.pkl")], dat.concept_names)

    random_state = 0
    cbe = {}
    cb_explainer = ConceptExplainer(dummy_prediction_model, dummy_representation_model, None,
                                    dat.get_test_dataset(), num_classes=dat.num_classes,
                                    concepts=dat.concept_names, random_state=random_state,
                                    custom_clip_model=dummy_clip_model)

    uace_wts, sigma = cb_explainer.uace(mode=NOISE_MODELLING.NONE)
    cbe['uace_none'] = (uace_wts, sigma)

    uace_wts, sigma = cb_explainer.uace()
    cbe['uace'] = (uace_wts, sigma)

    wts = cb_explainer.oikarinen_cbm(fit_intercept=False)
    cbe['ocbm'] = wts

    wts = cb_explainer.ycbm(fit_intercept=False, lasso_alpha=1e-3)
    cbe['ycbm'] = wts

    model_dir = "lightning_logs"
    with open(f"{model_dir}/simagenet_expts.pkl", "wb") as f:
        pickle.dump(cbe, f)

    for ci, class_idx in enumerate(dat.class_idxs):
        class_name = CLASS_NAMES[class_idx]
        print(f"\n------------------\nClass name: {class_name}")
        sidxs = np.argsort(uace_wts[ci])
        # print("Low scoring concepts")
        # for idx in sidxs[:10]:
        #     print(f"{dat.concept_names[idx]}: {wts[ci][idx]: 0.2f}")
        # print("\n\n")
        print("High scoring concepts")
        for idx in sidxs[-20:]:
            print(f"{dat.concept_names[idx]}: {uace_wts[ci][idx]: 0.2f}")
        print("\n\n")
