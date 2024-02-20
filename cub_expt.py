import pickle
import torch
from torchvision import models

from mydatasets.cub import CUB
from explanations.concept_explainers import ConceptExplainer
from explanations.tcav import Squeezer

import os
import pandas as pd
from clip import clip
from typing import List


if __name__ == '__main__':
    """
    Computes explanations for a model trained on CUB dataset using in-distribution probe-dataset: the test split.  
    """
    dat = CUB(fldr='/scratch/vp421/')
    cub_model = models.__dict__['resnet18'](num_classes=dat.num_classes)
    cub_model.load_state_dict(torch.load("lightning_logs/resnet18_cub_2011_200.pt"))
    cub_repr_model = torch.nn.Sequential(*list(cub_model.children())[:-1], Squeezer())
    random_state = 0
    cbe = {}
    cb_explainer = ConceptExplainer(cub_model, cub_repr_model, None, 
                                    dat.get_test_dataset(), num_classes=dat.num_classes, 
                                    concepts=dat.concept_names, random_state=random_state, 
                                    labels_of_interest=[_ for _ in range(200)])

#     print(dat.concept_names)
#     wts = cb_explainer.simple_fit()
#     cbe['simple'] = wts
#     # print(wts)

#     wts = cb_explainer.tcav(layer_index=-3)
#     cbe['tcav'] = wts
#     # print(wts)

#     wts, extra_obj = cb_explainer.oikarinen_cbm(fit_intercept=False, return_concept_reprs=True)
#     cbe['ocbm'] = wts
#     # print(wts)

#     wts = cb_explainer.ycbm(fit_intercept=False, lasso_alpha=1e-3)
#     cbe['ycbm'] = wts
#     # print(wts)

    model_dir = "lightning_logs"
    with open(f"{model_dir}/cub_expls.pkl", "rb") as f:
        cbe = pickle.load(f)
    wts, sigma = cb_explainer.uace(kappa=0.0)
    cbe['clip_cbe'] = (wts, sigma)
    print(wts, sigma)

    model_dir = "lightning_logs"
    with open(f"{model_dir}/cub_expls.pkl", "wb") as f:
        pickle.dump(cbe, f)
