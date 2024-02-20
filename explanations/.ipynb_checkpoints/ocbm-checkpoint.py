from sklearn.linear_model import LogisticRegression

from torchvision import transforms as T
from clip import clip
import torch
import torch.utils.data as data_utils
from tqdm import tqdm
import numpy as np

from mydatasets import cub

device = "cuda" if torch.cuda.is_available() else "cpu"

def cv_optimizer(context, train_reprs, train_P, val_reprs, val_P, use_cubic=True):
    """
    Optimizes and returns concept vectors that reside in the representation space of model-to-be-explained but
    apprimximate CLIP similarity scores.

    :param context:
    :param train_reprs:
    :param train_P:
    :param val_reprs:
    :param val_P:
    :param use_cubic:
    :return:
    """
    def sim_loss(p, q):
        # define the cubic cosine similarity of the paper
        assert len(p) == len(q), f"args are expected to be of same length but are of length: {len(p)}, {len(q)}"
        
        # if use_cubic:
        #     p, q = p ** 3, q ** 3
        num = (p * q).sum(dim=0)
        denom = (torch.norm(p, dim=0) * torch.norm(q, dim=0))
        dp = num/denom
        return -dp.sum()

    def csim(p, q):
        return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))

    repr_size, num_concepts = train_reprs.shape[-1], train_P.shape[-1]
    # repr_size x num_concepts
    torch.manual_seed(context.random_state)
    wc = torch.nn.Parameter(torch.randn([repr_size, num_concepts]))
    opt = torch.optim.Adam(lr=1e-1, params=[wc])
    max_steps = 2000
    best_wc, best_val_loss, best_si = None, None, None
    for si in tqdm(range(max_steps), desc="estimating wc matrix"):
        opt.zero_grad()
        train_proj_concepts = torch.mm(train_reprs, wc)
        train_loss = sim_loss(train_proj_concepts, train_P)
        train_loss.backward()
        opt.step()

        val_proj_concepts = torch.mm(val_reprs, wc)
        with torch.no_grad():
            val_loss = sim_loss(val_proj_concepts, val_P)

        val_loss = val_loss.detach().cpu().numpy()
        if not best_val_loss:
            best_val_loss = val_loss
            best_wc = wc.detach().cpu()
            best_si = si
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wc = wc.detach().cpu()
            best_si = si
        # if si % 100 == 0:
        #     print(train_loss.detach().cpu().numpy(), val_loss, wc.grad)

    train_proj_concepts = torch.mm(train_reprs, best_wc)
    # print("check: ", best_wc.sum(axis=0))
    print("-------------\ndebug")
    _proj_concepts = train_proj_concepts.detach().cpu().numpy()
    _P = train_P.numpy()
    cos_sims = []
    for ci in range(_P.shape[1]):
        _sim = csim(_proj_concepts[:, ci], _P[:, ci])
        # print(_proj_concepts[:20, ci], _P[:20, ci])
        # print(np.linalg.norm(_proj_concepts[:, ci]), np.linalg.norm(_P[:, ci]))
        # print(f"{ci}", _sim)
        cos_sims.append(_sim)

    print(f"Best checkpoint at step {best_si}, train loss: {train_loss.detach().cpu().numpy(): 0.3f}, "
          f"val loss: {best_val_loss: 0.3f}")

    return best_wc.detach(), cos_sims


class OCBM:
    @staticmethod
    def compute(context, return_acc, fit_intercept, return_concept_reprs, lasso_C=1):
        """
        Based on LABEL-FREE CONCEPT BOTTLENECK MODELS ("https://openreview.net/forum?id=FlCg47MNvBA")

        :param model:
        :param dataset:
        :param concepts:
        :return:
        """
        # ---------------
        # step 1: solve for a linear layer projecting representations to concept scores
        # ---------------        
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
                if type(batch_x) == torch.Tensor:
                    batch_x = batch_x.to(device)
                all_reprs.append(context.reprs_layer(batch_x).detach().cpu())
                if context.custom_clip_model:
                    logits_per_image = context.custom_clip_model(batch_x)
                else:
                    clip_batch = torch.stack([context.clip_preprocess(_pfn(_x)) for _x in batch_x],
                                             dim=0)
                    logits_per_image, logits_per_text = context.clip_model(clip_batch.to(device), clip_concepts.to(device))

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

        if isinstance(context.dataset, cub._Wrapper):
            # TODO: special handling needed here because using cubic is leading to 
            #       inf values in torch.norm(train_P**3, dim=0)
            wc, _ = cv_optimizer(context, train_reprs, train_P, val_reprs, val_P, use_cubic=False)
        else:
            wc, _ = cv_optimizer(context, train_reprs, train_P, val_reprs, val_P, use_cubic=True)

        # step 2: solve for linear projection from concepts to logits
        # N x num_concepts

        # X = torch.mm(all_reprs, wc).numpy()
        # y = all_y.numpy()

        # Get a normalized representation and make an indexed dataloader
        with torch.no_grad():
            X = torch.mm(all_reprs, wc).detach()
            y = all_y.detach()

            X_mean = torch.mean(X, dim=0, keepdim=True)
            X_std = torch.std(X, dim=0, keepdim=True)

            X -= X_mean
            X /= X_std

        if context.labels_of_interest:
            _idxs = []
            for _l in context.labels_of_interest:
                _idxs += list(np.where(y == _l)[0])
            _idxs = np.array(_idxs)
            label_to_idx = {_y: yi for yi, _y in enumerate(context.labels_of_interest)}
            X, y = X[_idxs], y[_idxs]
            y = torch.tensor([label_to_idx[_y] for _y in y.numpy()])
            assert len(np.unique(y)) == len(context.labels_of_interest), "some labels are missing in the training dataset"

        y = y.numpy()
        clf_ = LogisticRegression(fit_intercept=fit_intercept, penalty='l1', C=lasso_C, solver='saga',
                                  random_state=context.random_state)
        idxs = np.random.permutation(len(X))
        _ln = int(0.8 * len(idxs))
        train_idxs, val_idxs = idxs[:_ln], idxs[_ln:]
        train_X, train_y = X, y
        val_X, val_y = X[val_idxs], y[val_idxs]
        clf_.fit(train_X, train_y)
        acc = np.equal(clf_.predict(val_X), val_y).mean()
        print(f"Acc: {acc}")

        if context.num_classes == 2:
            # coef then is a 1 x num_concepts
            coef = np.stack([-clf_.coef_[0], clf_.coef_[0]], axis=0)
        else:
            coef = clf_.coef_
        return_extra = []
        if return_acc:
            return_extra.append(acc)
        if return_concept_reprs:
            return_extra.append(wc)
        if len(return_extra) > 0:
            return coef, return_extra
        else:
            return coef