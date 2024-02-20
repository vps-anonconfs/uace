import pickle

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from mydatasets.simple_tag_dataset import SimpleTag
from explanations.concept_explainers import ConceptExplainer, NOISE_MODELLING
from simple_trainer import two_layer_cnn, LitSimple


if __name__ == '__main__':
    """
    Same as tag_expt.py but with nuisance concepts added.
    """
    # courtesy Bard (prompt: "Give a python list of strings (without images) of 100 animal names"), isn't it awesome?!
    animal_names = ["lion", "tiger", "giraffe", "zebra", "monkey", "bear", "wolf", "fox", "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "deer", "rabbit", "raccoon", "squirrel", "mouse", "rat", "snake", "crocodile", "alligator", "turtle", "tortoise", "lizard", "chameleon", "iguana", "komodo dragon", "frog", "toad", "turtle", "tortoise", "leopard", "cheetah", "jaguar", "hyena", "wildebeest", "gnu", "bison", "antelope", "gazelle", "gemsbok", "oryx", "warthog", "hippopotamus", "rhinoceros", "elephant seal", "polar bear", "penguin", "flamingo", "ostrich", "emu", "cassowary", "kiwi", "koala", "wombat", "platypus", "echidna", "elephant"]

    tag_switch_prob = 0.5
    model_dir = f"lightning_logs/tag_dataset_switch-prob={tag_switch_prob}"
    checkpoint_name = f"{model_dir}/checkpoints/epoch=9-step=320.ckpt"
    random_state = 0
    dat = SimpleTag(tag_switch_prob=tag_switch_prob)

    net, reprs = two_layer_cnn(3, 96, width=4, num_classes=dat.num_classes)

    prediction_model = LitSimple(net, num_classes=dat.num_classes)
    with open(checkpoint_name, "rb") as f:
        prediction_model.load_state_dict(torch.load(f)['state_dict'], strict=True)

    cb_explainer = ConceptExplainer(net, reprs, net.final_layer, dat.get_test_dataset(),
                                num_classes=dat.num_classes,
                                concepts=dat.concept_names,
                                random_state=random_state)
    quantum = 2
    for ai in range(0, (len(animal_names)//quantum) + 1):
        cbe = {}
        dat.set_nuisance_concept_names(animal_names[:quantum*ai])
        print(animal_names[:quantum*ai], dat.concept_names)
        cb_explainer.set_concept_names(dat.concept_names)

        wts = cb_explainer.oikarinen_cbm()
        cbe['ocbm'] = wts
        print(wts)

        wts = cb_explainer.ycbm(lasso_alpha=1e-8)
        cbe['ycbm'] = wts
        print(wts)

        wts, sigma = cb_explainer.uace(mode=NOISE_MODELLING.MEDIUM)
        cbe['clip_cbe-no_input_noise'] = (wts, sigma)
        print(wts, sigma)

        wts, sigma = cb_explainer.uace(kappa=0)
        cbe['clip_cbe'] = (wts, sigma)
        print(wts, sigma)

        with open(f"{model_dir}/exptd_{quantum*ai}.pkl", "wb") as f:
            pickle.dump(cbe, f)