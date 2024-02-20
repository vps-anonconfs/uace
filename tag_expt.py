import pickle
import os

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from mydatasets.simple_tag_dataset import SimpleTag
from explanations.concept_explainers import ConceptExplainer, NOISE_MODELLING
from simple_trainer import two_layer_cnn, LitSimple


def _train(dat):
    net, reprs = two_layer_cnn(3, 96, width=4, num_classes=dat.num_classes)
    train_loader = torch.utils.data.DataLoader(dat.get_train_dataset(), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dat.get_test_dataset(), batch_size=32)

    model = LitSimple(net, dat.num_classes)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    
def get_importance(model, dataset1, dataset2):
    test_loader1 = torch.utils.data.DataLoader(dataset1, batch_size=32, shuffle=False)
    test_loader2 = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=False)
    logits1, logits2 = [], []
    acc1 = Accuracy(task='multiclass', num_classes=2)
    acc2 = Accuracy(task='multiclass', num_classes=2)
    for batch_x, batch_y, _ in test_loader1:
        logits1.append(model(batch_x).detach().cpu().numpy())
        pred_y = torch.argmax(model(batch_x), dim=-1).detach().cpu().numpy()
        acc1(model(batch_x), batch_y)
    
    for batch_x, batch_y, _ in test_loader2:
        logits2.append(model(batch_x).detach().cpu().numpy())
        acc2(model(batch_x), batch_y)
        
    logits1 = np.concatenate(logits1, axis=0)
    logits2 = np.concatenate(logits2, axis=0)
    # return np.mean((logits1 - logits2)/logits2, axis=0)
    return acc2.compute() - acc1.compute()


if __name__ == '__main__':
    tag_switch_prob = 0.
    model_dir = f"lightning_logs/tag_dataset_switch-prob={tag_switch_prob}"
    checkpoint_name = f"{model_dir}/checkpoints/epoch=9-step=320.ckpt"
    random_state = 0

    dat = SimpleTag(tag_switch_prob=tag_switch_prob)
    TRAIN = False
    if TRAIN:
        _train(dat)
        fldr = "lightning_logs/"
        # move the checkpoint
        os.system(f"ls -t {fldr}|head -n 1|xargs -I % mv {fldr}/% {fldr}/tag_dataset_switch-prob={tag_switch_prob}")

    net, reprs = two_layer_cnn(3, 96, width=4, num_classes=dat.num_classes)

    prediction_model = LitSimple(net, num_classes=dat.num_classes)
    with open(checkpoint_name, "rb") as f:
        prediction_model.load_state_dict(torch.load(f)['state_dict'], strict=True)
        
    cb_explainer = ConceptExplainer(net, reprs, net.final_layer, dat.get_test_dataset(),
                                    num_classes=dat.num_classes,
                                    concepts=dat.concept_names,
                                    random_state=random_state)

    cbe = {}
    # debug_dat = SimpleTag(add_tag=False)
    # debug_dat2 = SimpleTag(tag_switch_prob=tag_switch_prob, only_A=True)
    # debug_dat3 = SimpleTag(tag_switch_prob=tag_switch_prob, only_B=True)
    # ab_importance = get_importance(prediction_model.net, debug_dat.get_test_dataset(), dat.get_test_dataset())
    # b_importance = get_importance(prediction_model.net, debug_dat2.get_test_dataset(), dat.get_test_dataset())
    # a_importance = get_importance(prediction_model.net, debug_dat3.get_test_dataset(), dat.get_test_dataset())
    # print(ab_importance, b_importance, a_importance)
    
#     cbe['gt'] = {'a': a_importance, 'b': b_importance, 'ab': ab_importance, '~ab':1-ab_importance}
    wts = cb_explainer.simple_fit(lasso_alpha=1e-6)
    cbe['simple'] = wts
    print(wts)

    wts, sigma = cb_explainer.bayesian_fit()
    cbe['bayesian'] = (wts, sigma)
    print(wts, sigma)

    wts = cb_explainer.tcav(layer_index=-2)
    cbe['tcav'] = wts
    print(wts)

    wts = cb_explainer.oikarinen_cbm()
    cbe['ocbm'] = wts
    print(wts)

    # wts, sigma = cb_explainer.clip_cbe(mode=NOISE_MODELLING.NONE)
    wts = cb_explainer.ycbm()
    cbe['ycbm'] = wts
    print(wts)
        
    wts, sigma = cb_explainer.uace(kappa=0.000)
    cbe['clip_cbe'] = (wts, sigma)
    print(wts, sigma)

    with open(f"{model_dir}/eval.pkl", "wb") as f:
        pickle.dump(cbe, f)
