from sklearn.linear_model import Lasso
from torchvision import transforms as T
from clip import clip
import torch
import torch.utils.data as data_utils
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class YCBM:
    @staticmethod
    def compute(context, return_acc, fit_intercept, lasso_alpha):
        """
        Based on Posthoc CBMs 

        :param model:
        :param dataset:
        :param concepts:
        :return:
        """
        dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)

        # precompute P and reprs
        assert context.concepts and len(context.concepts) > 0, \
            f"Concepts param should be set to non-empty when using this explainer"
        clip_concepts = clip.tokenize(context.concepts).to(device)

        all_reprs, all_logits, all_clip_sims, all_y, all_g = [], [], [], [], []
        _pfn = T.ToPILImage()
        for batch_x, batch_y, batch_g in tqdm(dl, desc="Getting CLIP sims"):
            with torch.no_grad():
                if type(batch_x) == torch.Tensor:
                    batch_x = batch_x.to(device)
                all_reprs.append(context.reprs_layer(batch_x).detach().cpu())
                all_logits.append(context.model(batch_x).detach().cpu())
                if context.custom_clip_model:
                    logits_per_image = context.custom_clip_model(batch_x)
                else:
                    clip_batch = torch.stack([context.clip_preprocess(_pfn(_x)) for _x in batch_x],
                                             dim=0)
                    logits_per_image, logits_per_text = context.clip_model(clip_batch.to(device), clip_concepts.to(device))

            _clip_sims = logits_per_image.detach().cpu() / 100
            all_clip_sims.append(_clip_sims)
            all_y.append(batch_y.cpu().numpy())
            all_g.append(batch_g.cpu())

        X = np.concatenate(all_clip_sims, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        print("Check:", type(X), X.shape)
        if context.labels_of_interest:
            _idxs = []
            for _l in context.labels_of_interest:
                _idxs += list(np.where(all_y == _l)[0])
            _idxs = np.array(_idxs)
            label_to_idx = {_y: yi for yi, _y in enumerate(context.labels_of_interest)}

            X, all_logits = X[_idxs], all_logits[_idxs]
            all_logits = all_logits[:, np.array(context.labels_of_interest)]
            # y = np.array([label_to_idx[_y] for _y in all_y])
            # assert len(np.unique(all_y)) == len(context.labels_of_interest), "some labels are missing in the training dataset"

        # y need not be continuous, it works even otherwise
        rng = np.random.default_rng(context.random_state)
        _idxs = rng.permuted(np.arange(len(X)))
        _ln = int(len(_idxs) * 0.8)
        train_idxs, val_idxs = _idxs[:_ln], _idxs[_ln:]

        train_X, val_X = X, X[val_idxs]
        train_Y_, val_Y_ = all_logits, all_logits[val_idxs]
        num_classes = all_logits.shape[-1]
        all_coefs = []
        for ci in range(num_classes):
            fitter = Lasso(alpha=lasso_alpha, fit_intercept=True, random_state=context.random_state)
            fitter.fit(X, all_logits[:, ci])
            all_coefs.append(fitter.coef_)

        """
        Posthoc-CBM proposed to fit coefficients that regress the true labels. 
        That is replaced by regression to logits above. 
        """
        # best_score = -1e10
        # for _ in range(5):
        #     alpha, l1_ratio = 10**np.random.uniform(-1, 1), np.random.uniform(1e-3, 1-1e-3)
        #     clf_ = LogisticRegression(fit_intercept=fit_intercept, penalty='elasticnet', solver='saga', l1_ratio=l1_ratio, C=alpha)
        #     clf_.fit(train_X, train_Y_)
        #     wt = clf_.coef_
        #     this_score = clf_.score(val_X, val_Y_)
        #     if this_score > best_score:
        #         best_wt, best_score = clf_.coef_, this_score
        #         best_vals = (alpha, l1_ratio)
        #         best_acc = ((clf_.predict(train_X) == train_Y_).mean(), (clf_.predict(val_X) == val_Y_).mean())
        # wt = best_wt
        # print(f"Best score: {best_score}, best params: {best_vals} best acc: {best_acc}")

        # if context.num_classes == 2:
        #     # coef then is a 1 x num_concepts
        #     coef = np.stack([-wt[0], wt[0]], axis=0)
        # else:
        #     coef = clf_.coef_

        return all_coefs