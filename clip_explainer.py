import matplotlib.pyplot as plt
import torchvision
from torch.utils import data as data_utils
import numpy as np
import torch
from torchvision import transforms as T
import clip
from typing import List
import tqdm
import pytorch_lightning as pl

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, BayesianRidge

import simple_trainer
from mydatasets.colorj import ColorJ
from mydatasets.mnist_juxtaposed import MNISTJ
from mydatasets.simple_tag_dataset import SimpleTag

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

class CLIPWrapper(data_utils.Dataset):
    def __init__(self, dat, clip_preprocess):
        self.dataset = dat
        self.preprocess = clip_preprocess
        self.pre_preprocess = T.ToPILImage()

    def __getitem__(self, item):
        img, y = self.dataset[item]
        return self.preprocess(self.pre_preprocess(img)), y

    def __len__(self):
        return len(self.dataset)


def logit_fit(X, y):
    """
    :param X: a two-dimensional input feature matrix [num_examples, num_features]
    :param y: logit values of size [num_examples] where each value is a probability value in range [0, 1]
    :return: a weight vector of size [num_features] that explains the logits well
    """
    num_examples, num_features = X.shape
    wt = torch.nn.Parameter(torch.rand([num_features])*1e-5, requires_grad=True)
    bias = torch.nn.Parameter(torch.zeros([]), requires_grad=True)
    opt = torch.optim.Adam(lr=1e-2, params=[wt, bias], weight_decay=1)
    _X = torch.tensor(X)
    for _ in range(1000):
        opt.zero_grad()
        logits = torch.matmul(_X, wt.unsqueeze(dim=-1)) + bias
        predicted_probs = torch.squeeze(torch.sigmoid(logits))
        loss = -(y * torch.log(predicted_probs + 1e-5) + (1 - y)*torch.log(1 - predicted_probs + 1e-5)).mean()
        if _ % 50 == 0:
            print(loss.detach())
        loss.backward()
        opt.step()

    return wt.detach().cpu().numpy()


def global_explain(blackbox_model, train_dataset: data_utils.Dataset, concepts: List[str]) -> np.array:
    """
    Global explanation model using clip
    Fits a logistic classifier on entire training data to predict the prediction probabilities using clip's token
    similarities
    :param blackbox_model:
    :param train_dataset:
    :param concepts:
    :return: an array of same length as concepts such that the absolute value captures the importance of the
    corresponding concept in explaining the predictions of the blackbox model.
    """
    clip_concepts = clip.tokenize(concepts).to(device)

    loader = data_utils.DataLoader(train_dataset, batch_size=32, shuffle=False)
    clip_loader = data_utils.DataLoader(CLIPWrapper(train_dataset, clip_preprocess), batch_size=32, shuffle=False)
    bb_probs, clip_out = [], []
    with torch.no_grad():
        num_corr = 0
        for batch_x, batch_y in tqdm.tqdm(loader, desc="Fetching bb predictions"):
            _logits = blackbox_model(batch_x)
            _probs = torch.softmax(_logits, dim=-1)
            bb_probs.append(_probs)
            _preds = torch.argmax(_probs, dim=-1)
            num_corr += torch.sum((_preds == batch_y).type(torch.float32))

        print(f"Accuracy: {num_corr/len(train_dataset)}")

        for clip_batch, y in tqdm.tqdm(clip_loader, desc="Fetching clip predictions"):
            logits_per_image, logits_per_text = clip_model(clip_batch, clip_concepts)
            _out = logits_per_image.cpu().numpy()/100
            clip_out.append(_out)
    bb_probs, clip_out = np.concatenate(bb_probs, axis=0), np.concatenate(clip_out, axis=0)
    # for _i in range(30):
    #     print(bb_probs[_i], clip_out[_i])

    # fit a simple linear model to predict bb model outputs
    # fitter = Ridge(alpha=.1, fit_intercept=True)
    fitter = BayesianRidge(fit_intercept=True)
    fitter.fit(clip_out, bb_probs[:, 0])

    k = len(concepts)
    # print("Sigma", fitter.sigma_[np.arange(k), np.arange(k)])
    #
    # for ci in range(clip_out.shape[1]):
    #     print(f"{ci}: {np.corrcoef(clip_out[:, ci], bb_probs[:, 0])[1, 0]}")
    return fitter.coef_


def local_explain(prediction_model, example, concepts):
    clip_concepts = clip.tokenize(concepts).to(device)

    _t = T.ToPILImage()
    unif_distr = torch.distributions.Uniform(low=0, high=0.5)
    blur_transform = torchvision.transforms.GaussianBlur(3, sigma=(0.01, 10))
    auto_transform = torchvision.transforms.RandAugment()
    # auto_transform = torchvision.transforms.AugMix(severity=3)
    # AugMix()
    num_samples = 64
    pre_preprocess = T.ToPILImage()
    with torch.no_grad():
        # batch_x = torch.stack([example] + [blur_transform(example) for _ in range(num_samples)], dim=0)
        # batch_x = example.unsqueeze(dim=0) + unif_distr.rsample(torch.Size([num_samples]) + example.shape)
        _example = T.functional.convert_image_dtype(example, dtype=torch.uint8)
        batch_x = torch.stack([_example] + [auto_transform.forward(_example) for _ in range(num_samples)], dim=0)
        batch_x = T.functional.convert_image_dtype(batch_x, dtype=torch.float32)
        _logits = prediction_model(batch_x)
        _probs = _logits

        clip_batch = torch.stack([clip_preprocess(pre_preprocess(_x)) for _x in batch_x], dim=0)
        logits_per_image, logits_per_text = clip_model(clip_batch, clip_concepts)
        # _clip_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        _clip_probs = logits_per_image.cpu().numpy()/100

        # _probs = _probs[1:, ] - _probs[0]
        # _clip_probs = _clip_probs[1:, ] - _clip_probs[0]
        # for _i in range(len(batch_x)):
        #     img = batch_x[_i]
        #     clip_p = _clip_probs[_i]
        #     plt.imshow(torch.permute(img, [1, 2, 0]))
        #
        #     info = f"{_probs[_i][0]} clip: {list(zip(clip_p, concepts))}"
        #     plt.title(info)
        #     print(info)
        #     plt.draw()
        #     if plt.waitforbuttonpress():
        #         next

    # pred_y = torch.argmax(_logits[0])
    pred_y = 0 # torch.argmax(_logits[0])
    # print(_clip_probs, _probs[:, pred_y], pred_y, _logits)
    # for ci in range(len(_clip_probs)):
    #     print(_clip_probs[ci], _logits[ci])
    # when alpha is set to 0, the solution is very unstable. We get very poor R2 then.
    print("Pred:", pred_y)
    if True:
        # fitter = Ridge(alpha=0.1, fit_intercept=True)
        fitter = BayesianRidge(fit_intercept=True)
        X, y = _clip_probs, _probs[:, pred_y]
        fitter.fit(X, y)
        wt = fitter.coef_
        # wt /= fitter.intercept_
        print("Intercept: ", fitter.intercept_)
        k = len(concepts)
        print("Sigma", fitter.sigma_[np.arange(k), np.arange(k)])
        print("R2: ", fitter.score(X, y))
        # wt = logit_fit(_clip_probs, _probs[:, pred_y])
    else:
        all_wts = []
        for _i in range(len(batch_x)//16):
            fitter = Ridge(alpha=0.1, fit_intercept=True)
            # fitter = BayesianRidge(fit_intercept=True)
            X, y = _clip_probs[_i*16: (_i+1)*16], _probs[_i*16: (_i+1)*16, pred_y]
            fitter.fit(X, y)
            wt = fitter.coef_
            all_wts.append(wt)
        all_wts = np.stack(all_wts, axis=0)
        wt = np.mean(all_wts, axis=0)
        sigma = np.std(all_wts, axis=0)
        print("Sigma:", sigma)
    return wt


if __name__ == '__main__':
    tag_switch_prob = 0

    # checkpoint_name = f"lightning_logs/tag_switch_prob={tag_switch_prob}/checkpoints/epoch=9-step=320.ckpt"
    # dat = SimpleTag(tag_switch_prob)
    # checkpoint_name = f"lightning_logs/colo_train-prob=0.95/checkpoints/epoch=9-step=1570.ckpt"
    checkpoint_name = f"lightning_logs/colorj_train-prob=0.5/checkpoints/epoch=99-step=400.ckpt"

    random_state = 0
    dat = ColorJ(train_rand_prob=0.5, test_rand_prob=0.5)
    net, reprs = simple_trainer.two_layer_cnn(3, 56, width=4, num_classes=dat.num_classes)
    prediction_model = simple_trainer.LitSimple(net, num_classes=dat.num_classes)
    with open(checkpoint_name, "rb") as f:
        prediction_model.load_state_dict(torch.load(f)['state_dict'], strict=True)

    # test_loader = data_utils.DataLoader(dat.get_test_dataset())
    # trainer = pl.Trainer(max_epochs=10)
    # trainer.test(prediction_model, test_loader)
    # concept_tokens = ["car", "wheel", "window", "plane", "wing", "fin", "a", "b", "dog"]
    # concept_tokens = ["body", "headlights", "taillights", "turn signals", "windshield", "windshield vipers", "bumpers", "wheels", "tires"]
    # concept_tokens += ["fuselage", "wings", "tail assembly", "landing gear", "engines"]
    # concept_tokens += ["a", "b", "dog"]
    concept_tokens = dat.concept_names
    # example_idx = 26
    all_wts = []
    for example_idx in range(25):
        # example, y = dat.get_test_dataset(tag_prob=tag_switch_prob)[example_idx]
        example, y, g = dat.get_test_dataset()[example_idx]
        wt = local_explain(prediction_model.net, example=example, concepts=concept_tokens)
        all_wts.append(wt)
        print(wt)
        # print(f"y: {y}")
        # plt.imshow(torch.permute(example, [1, 2, 0]))
        # plt.show()
    mean_wt = np.stack(all_wts, axis=0).mean(axis=0)
    print(mean_wt)
    # 50
    # 0.5 :: [ 0.39566255 -0.0859845  -0.04418699 -0.13841419 -0.07367393 -0.05340282 ]
    # 0 :: [ 0.00265875 -0.00016191 -0.00135578 -0.00066639 -0.00026918 -0.00020549 ]
    # 12
    # 0 :: [ 8.3675641e-06 -1.5e-07 -3.6131660e-06 -4.0e-06 1.1e-07 -6.7e-07 ]
    # 0.5 :: [-0.05522051 -0.18583485  0.00733801  0.14674819  0.01184497  0.07512369]
    # 0
    # [ 0.00231812 -0.04892407 -0.15136363  0.03298618 -0.06063102 -0.03976518
    #  -0.19697066 -0.0677774   0.09543523  0.06408419  0.04920572 -0.00026563
    #   0.06908202  0.09962627  0.13104935 -0.02732495 -0.01476502]
    # 0.5
    # [0.0121216   0.00620684 - 0.05586221 - 0.01042308 - 0.03053022 - 0.02414125
    #  - 0.0121516 - 0.00101085 - 0.04304573 - 0.00236257  0.00503495  0.00292851
    #  0.05107922  0.00323256  0.02703792 - 0.02429817 - 0.00345766]
    # [-0.009118 - 0.00578375 - 0.01211813 - 0.008363 - 0.01195995 - 0.0215565
    #  - 0.0160049 - 0.00586065 - 0.01598749  0.03409137 - 0.00376829 - 0.01092147
    #  0.01961327  0.00397632 - 0.00063703 - 0.0093664 - 0.01688705]

    # _dat = dat.get_train_dataset()
    # _dat = dat.get_test_dataset(tag_prob=tag_switch_prob)
    # rng = np.random.default_rng(0)
    # # idxs = rng.choice(len(_dat), size=[200])
    # idxs = list(range(30))
    # _dat = torch.utils.data.Subset(_dat, idxs)
    # wt = global_explain(prediction_model.net, train_dataset=_dat, concepts=concept_tokens)
    # print(wt)
