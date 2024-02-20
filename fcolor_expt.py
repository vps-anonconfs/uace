import pickle

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from mydatasets.four_color import FColor
from explanations.concept_explainers import ConceptExplainer, NOISE_MODELLING
from simple_trainer import two_layer_cnn, LitSimple


def _train(dat):
    net, reprs = two_layer_cnn(3, 28, width=4, num_classes=dat.num_classes)
    train_loader = torch.utils.data.DataLoader(dat.get_train_dataset(), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dat.get_test_dataset(), batch_size=32)

    model = LitSimple(net, dat.num_classes)

    trainer = pl.Trainer(max_epochs=30)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)


def compute_gt_scores(net, random_state):
    dat_eval = FColor(test_mode_mix=0.5, num_test=500, random_state=random_state)
    per_concept_acc, accs = [], []
    num_concepts = 4
    for ci in range(num_concepts):
        corrs1, corrs2 = [], []
        for batch_x, batch_y, batch_g in DataLoader(dat_eval.get_test_dataset(), batch_size=32, shuffle=True):
            batch_size = len(batch_x)
            _mn = torch.mean(batch_x.view([batch_size, -1]), dim=-1)

            iidxs = torch.where(batch_g[:, ci] == 1)[0]
            if len(iidxs) == 0:
                continue
            probs = net(batch_x[iidxs])
            corrs1.append(probs[:, 0])
            corrs2.append(probs[:, 1])

        acc1 = torch.cat(corrs1, dim=0).type(torch.float32).mean()
        acc2 = torch.cat(corrs2, dim=0).type(torch.float32).mean()
        if ci < 2:
            accs.append(acc1)
        else:
            accs.append(acc2)
        print(f"Concept: {ci}, acc: {accs[-1]}")

    return accs


if __name__ == '__main__':
    model_dir = "lightning_logs/fcolor"
    mode_mix = 1.0
    cbe = {}
    random_state = 42
    dat = FColor(test_mode_mix=mode_mix, num_test=100)

    TRAIN = False

    if TRAIN:
        # do not forget to move the checkpoint to appropriate folder
        _train(dat)

    checkpoint_name = f"{model_dir}/checkpoints/epoch=29-step=960.ckpt"
    net, reprs = two_layer_cnn(3, 28, width=4, num_classes=dat.num_classes)
    prediction_model = LitSimple(net, num_classes=dat.num_classes)
    with open(checkpoint_name, "rb") as f:
        prediction_model.load_state_dict(torch.load(f)['state_dict'], strict=True)

    gt_accs = compute_gt_scores(net, random_state)
    cbe['gt'] = gt_accs

    cb_explainer = ConceptExplainer(net, reprs, net.final_layer, dat.get_test_dataset(),
                                    num_classes=dat.num_classes,
                                    concepts=dat.concept_names,
                                    random_state=random_state)

    wts = cb_explainer.simple_fit()
    cbe['simple'] = wts
    print(wts)

    wts, sigma = cb_explainer.bayesian_fit()
    cbe['bayesian'] = (wts, sigma)
    print(wts, sigma)

    wts = cb_explainer.tcav()
    cbe['tcav'] = wts
    print(wts)

    wts = cb_explainer.oikarinen_cbm()
    cbe['ocbm'] = wts
    print(wts)

    wts, sigma = cb_explainer.uace(mode=NOISE_MODELLING.NONE)
    cbe['ycbm'] = (wts, sigma)
    print(wts, sigma)
    
    wts, sigma = cb_explainer.uace(mode=NOISE_MODELLING.MEDIUM)
    cbe['clip_cbe-no_input_noise'] = (wts, sigma)
    print(wts, sigma)
    
    wts, sigma = cb_explainer.uace()
    cbe['clip_cbe'] = (wts, sigma)
    print(wts, sigma)

    with open(f"{model_dir}/eval-on-testm={mode_mix}.pkl", "wb") as f:
        pickle.dump(cbe, f)
