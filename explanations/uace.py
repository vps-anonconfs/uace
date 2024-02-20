from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from torchvision import transforms as T
from clip import clip
import torch
import torch.utils.data as data_utils
from tqdm import tqdm
import numpy as np

import pyro
from pyro import distributions as dist
import pyro.distributions.constraints as constraints
from enum import Enum

from explanations.ocbm import cv_optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class NOISE_MODELLING(Enum):
    NONE = 0
    MEDIUM = 1
    FULL = 2


class UACE:
    @staticmethod
    def compute(context, mode, return_acc, kappa):
        """
        kappa: sparsify such that the accuracy does not far by more than kappa value
        """
        rng = np.random.default_rng(context.random_state)
        dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)

        # precompute P and reprs
        assert context.concepts and len(context.concepts) > 0, \
            f"Concepts param should be set to non-empty when using this explainer"
        clip_concepts = clip.tokenize(context.concepts).to(device)

        all_logits, all_clip_sims, all_y, all_g = [], [], [], []
        all_reprs = []
        _pfn = T.ToPILImage()

        for batch_x, batch_y, batch_g in tqdm(dl, desc="Computing clip sims"):
            if type(batch_x) == torch.Tensor:
                batch_x = batch_x.to(device)
            with torch.no_grad():
                all_logits.append(context.model(batch_x).detach().cpu().numpy())
                if context.custom_clip_model:
                    clip_logits_per_image = context.custom_clip_model(batch_x)
                else:
                    clip_batch = torch.stack([context.clip_preprocess(_pfn(_x)) for _x in batch_x], dim=0)
                    clip_logits_per_image, clip_logits_per_text = context.clip_model(clip_batch.to(device),
                                                                                  clip_concepts.to(device))
            _clip_sims = clip_logits_per_image.detach().cpu() / 100
            all_reprs.append(context.reprs_layer(batch_x).detach().cpu().numpy())
            all_clip_sims.append(_clip_sims)
            all_y.append(batch_y.cpu())
            all_g.append(batch_g.cpu())

        all_clip_sims = np.concatenate(all_clip_sims, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        all_g = np.concatenate(all_g, axis=0)
        all_reprs = np.concatenate(all_reprs, axis=0)
        all_y = np.concatenate(all_y)
        num_concepts = all_clip_sims.shape[-1]

        original_logits = all_logits.copy()
        all_logits = StandardScaler(with_mean=False).fit_transform(all_logits)

        # print(all_clip_sims.shape, all_g.shape)
        # for ci in range(num_concepts):
        #     print(ci, " -- ", np.corrcoef(all_clip_sims[:, ci], all_g[:, ci])[1, 0])

        # estimate noise level per concept
        P, all_reprs = torch.tensor(all_clip_sims), torch.tensor(all_reprs)
        mu, sigma = torch.mean(P, dim=0).unsqueeze(dim=0), torch.std(P, dim=0).unsqueeze(dim=0)
        P = (P - mu) / sigma
        _idxs = rng.permuted(np.arange(len(context.dataset)))
        _ln = int(len(_idxs) * 0.8)
        train_idxs, val_idxs = _idxs[:_ln], _idxs[_ln:]
        train_reprs, train_P = all_reprs[train_idxs, :], P[train_idxs, :]
        val_reprs, val_P = all_reprs[val_idxs, :], P[val_idxs, :]
        wc, noise_per_concept = cv_optimizer(context, train_reprs, train_P, val_reprs, val_P, use_cubic=False)

        # TODO: ideally the noise_per_concept estimation should account for estimation error.
        num_labels = all_logits.shape[-1]
        if context.labels_of_interest:
            labels = context.labels_of_interest
            num_labels = len(labels)
        else:
            labels = list(range(num_labels))

        num_labels = all_logits.shape[-1]
        wts, sigmas = [], []
        if context.labels_of_interest:
            idxs_of_interest = []
            for _l in context.labels_of_interest:
                idxs_of_interest += list(np.where(all_y == _l)[0])
            idxs_of_interest = np.array(idxs_of_interest)
        else:
            idxs_of_interest = np.arange(len(all_logits))
        _idxs, _ln = rng.permuted(idxs_of_interest), int(0.8 * len(idxs_of_interest))
        train_idxs, val_idxs = _idxs[:_ln], _idxs[_ln:]
        Y, val_Y = all_logits, all_logits[val_idxs]
        X, val_X = all_clip_sims, all_clip_sims[val_idxs]

        if (mode == NOISE_MODELLING.FULL) or (mode == NOISE_MODELLING.MEDIUM):
            # if the estimated noise is negative, it is an indication of poor estimation
            # because even a random vector on average has 0 cosine similarity, so we clip to 0.
            noise_per_concept = np.clip(noise_per_concept, a_min=0, a_max=None)
            X = X * np.reshape(noise_per_concept, [1, -1])

        # hyperparam caching for efficiency
        beta, gamma = None, None
        for yi in tqdm(labels, desc="Iteration"):
            if mode == NOISE_MODELLING.NONE:                
                clf_ = BayesianRidge(fit_intercept=True)
                clf_.fit(X, Y[:, yi])
                wt, sigma = clf_.coef_, np.diag(clf_.sigma_)
            elif mode == NOISE_MODELLING.MEDIUM:
                # N x c
                sin_theta = np.sin(np.arccos(all_clip_sims))
                # 1 x c
                sin_alpha = np.reshape(np.sin(np.arccos(noise_per_concept)), [1, -1])
                # num_concepts size vector
                eps = np.mean(np.abs(sin_theta * sin_alpha), axis=0)
                
                num_samples = 10
                sampled_x, sampled_y = [], []
                for si in range(num_samples):
                    _eps = eps[None, :]
                    sampled_x.append(X + np.random.uniform(-_eps, _eps))
                    sampled_y.append(Y[:, yi])
                sampled_x = np.concatenate(sampled_x, axis=0)
                sampled_y = np.concatenate(sampled_y, axis=0)
                
                clf_ = BayesianRidge(fit_intercept=True)
                clf_.fit(sampled_x, sampled_y)
                wt, sigma = clf_.coef_, np.diag(clf_.sigma_)
            else:
                # N x c
                sin_theta = np.sin(np.arccos(all_clip_sims))
                # 1 x c
                sin_alpha = np.reshape(np.sin(np.arccos(noise_per_concept)), [1, -1])
                # num_concepts size vector
                # eps = np.sqrt(((sin_theta*sin_alpha)**2).sum(axis=0))
                eps = np.mean(np.abs(sin_theta * sin_alpha), axis=0)
                # wt, sigma = my_bayes_estimate(X, Y[:, yi], eps=eps, beta_=1, lambda1=10, lambda2=0, fit_intercept=True)
                # wt, sigma = my_bayes_estimate2(X, Y[:, yi], err_bounds=np.abs(sin_theta*sin_alpha), fit_intercept=True)

                wt, sigma, beta, gamma = my_bayes_estimate3(X, Y[:, yi], beta, gamma, eps=eps, fit_intercept=False)
                # print("Per concept err: ", eps)

            wts.append(wt)
            sigmas.append(sigma)

        mean = np.stack(wts, axis=0)
        sigmas = np.stack(sigmas, axis=0)

        # sparsify weights
        if mode == NOISE_MODELLING.FULL:
            X = all_clip_sims - np.expand_dims(np.mean(all_clip_sims, axis=0), axis=0)
            val_X = val_X - np.expand_dims(np.mean(all_clip_sims, axis=0), axis=0)
            print(
                f"Debug: shape of wts: {mean.shape}, shape of all_logits: {all_logits.shape, all_logits[:, np.array(labels)].shape}, shape of X: {X.shape}")

            def _acc(mean, debug=False):
                surr_preds = np.argmax(np.matmul(val_X, np.transpose(mean)), axis=-1)
                orig_preds = np.argmax(val_Y[:, np.array(labels)], axis=-1)
                if debug:
                    print(surr_preds[:10], orig_preds[:10], np.mean(surr_preds[:10] == orig_preds[:10]))
                    print(all_logits[:10, np.array(labels)], np.matmul(X[:10], np.transpose(mean)))
                return np.mean(surr_preds == orig_preds)

            thresholds = np.quantile(np.abs(mean), np.arange(0, 1, 0.001))
            original_acc = _acc(mean)
            for ti in range(len(thresholds)):
                new_mean = np.where(np.abs(mean) < thresholds[ti], np.zeros_like(mean), mean)
                new_acc = _acc(new_mean)
                if new_acc <= (original_acc - kappa):
                    break
            mean = np.where(np.abs(mean) < thresholds[max(0, ti - 1)], np.zeros_like(mean), mean)
            print(f"Threshold: {ti}, {thresholds[ti]}, original: {original_acc}, new: {_acc(mean, debug=True)}")

        if return_acc:
            return mean / sigmas, sigmas, _acc(mean)
        return mean / sigmas, sigmas


def my_bayes_estimate(X, y, eps, beta_=1, lambda1=1, lambda2=1, fit_intercept=True):
    """
    :param X: must be N x num_features np matrix
    :param y: N size numpy vector
    :param eps: diag of prior Sigma on w
    :param beta_: precision on label noise
    :param lambda1: strength of prior1
    :param lambda2: strength of prior2
    """
    # X = StandardScaler().fit_transform(X)
    X_mu = np.expand_dims(X.mean(axis=0), axis=0)
    X = X - X_mu

    if fit_intercept:
        X = np.concatenate([X, np.ones([len(X), 1])], axis=1)
        # some non-informative prior on intercept
        eps = np.insert(eps, len(eps), 10)
    S0 = np.diag(eps ** 2)
    # assumed to be always 0
    # m0 = np.zeros_like(eps)

    SN_inv = lambda1 * np.linalg.inv(S0) + beta_ * np.matmul(X.transpose(), X) + lambda2 * np.eye(len(S0))
    SN = np.linalg.pinv(SN_inv)

    mN = np.matmul(SN, beta_ * np.matmul(X.transpose(), y))

    mean, sigma = mN, np.diag(SN)
    if fit_intercept:
        mean, sigma = mean[:-1], sigma[:-1]
    return mean, sigma

def my_bayes_estimate2(X, y, err_bounds, fit_intercept=True):
    def simple_model(data_x, errs, data_y=None):
        y_noise = pyro.sample("y_noise", dist.Uniform(0, 10)).to(X.device)
        num_dim = data_x.shape[-1]
        w = pyro.sample("w", dist.MultivariateNormal(torch.ones([num_dim]), torch.eye(num_dim))).type(X.dtype).to(
            X.device)

        _eps = torch.ones_like(X)
        noise_distr = dist.Uniform(-_eps, _eps).to_event(1)
        batch_size = 128
        with pyro.plate("data", len(data_x), subsample_size=batch_size, device=X.device) as ind:
            inp_noise = noise_distr.sample()
            perturbed_data = data_x.index_select(0, ind) + inp_noise.index_select(0, ind) * err_bounds.index_select(
                0,
                ind)
            mu = torch.einsum("Ni,i->N", perturbed_data, w)
            return pyro.sample("obs", pyro.distributions.Normal(mu, y_noise), obs=data_y.index_select(0, ind))

    if fit_intercept:
        X = np.concatenate([X, np.ones([len(X), 1])], axis=-1)
        err_bounds = np.concatenate([err_bounds, np.zeros([len(err_bounds), 1])], axis=-1)

    X, y = torch.tensor(X), torch.tensor(y)
    err_bounds = torch.tensor(err_bounds)
    X, y, err_bounds = X.to(device), y.to(device), err_bounds.to(device)
    pyro.clear_param_store()

    adam = pyro.optim.Adam({"lr": 1e-2})
    elbo = pyro.infer.Trace_ELBO()
    auto_guide = pyro.infer.autoguide.AutoNormal(simple_model)
    svi = pyro.infer.SVI(simple_model, auto_guide, adam, elbo)

    losses = []
    for step in range(1000):
        loss = svi.step(X, err_bounds, y)
        losses.append(loss)
        if step % 100 == 0:
            print("Elbo loss: {}".format(loss))

    w = pyro.param("AutoNormal.locs.w").data.cpu().numpy()
    scale = pyro.param("AutoNormal.scales.w").data.cpu().numpy()

    return w, scale

def my_bayes_estimate3(X, y, beta, gamma, eps, fit_intercept=True):
    """
    Same as my_bayes_estimate but all params obtained using maximum likelihood estimation
    :param X: must be N x num_features np matrix
    :param y: N size numpy vector
    :param eps: diag of prior Sigma on w
    :param beta_: precision on label noise
    :param lambda1: strength of prior1
    :param lambda2: strength of prior2

    https://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
    """

    # assert fit_intercept==False, "Not sure if we are handling the intercept right, fit_intercept should be set to false"

    def simple_model(data_x, data_y=None):
        y_noise = pyro.sample("y_noise", dist.Gamma(1., 1.)).to(data_x.device)
        num_dim = data_x.shape[-1]
        w = pyro.sample("w", dist.MultivariateNormal(torch.zeros([num_dim]).to(data_x.device),
                                                     torch.eye(num_dim).to(data_x.device))).type(data_x.dtype)

        _eps = eps.view([1, len(eps)]).repeat([len(data_x), 1])
        noise_distr = dist.Uniform(-_eps, _eps).to_event(1)
        batch_size = min(128, len(data_x))
        with pyro.plate("data", len(data_x), subsample_size=batch_size, device=data_x.device) as ind:
            inp_noise = noise_distr.sample().type(data_x.dtype)
            perturbed_data = data_x.index_select(0, ind) + inp_noise.index_select(0, ind)
            mu = torch.einsum("Ni,i->N", perturbed_data, w)
            return pyro.sample("obs", pyro.distributions.Normal(mu, y_noise), obs=data_y.index_select(0, ind))

    def estimate_with(gamma, beta_, data_x, data_y):
        # S0_inv = gamma * torch.diag(1. / eps ** 2)
        S0_inv = gamma * torch.diag(eps ** 2)
        SN_inv = S0_inv + beta_ * torch.mm(data_x.t(), data_x)
        SN = torch.linalg.pinv(SN_inv)
        # computes A.pinv() @ B
        mN = torch.linalg.lstsq(SN_inv, beta_ * torch.mv(data_x.t(), data_y)).solution
        return mN, SN, SN_inv

    def custom_guide(data_x, data_y=None):
        num_dim = data_x.shape[-1]

        gamma = pyro.param("gamma", init_tensor=torch.tensor(1), constraint=constraints.positive).to(data_x.device)
        beta_ = pyro.param("beta", init_tensor=torch.tensor(1), constraint=constraints.positive).to(data_x.device)
        y_noise = pyro.sample("y_noise", dist.Normal(1. / beta_, 1e-5))

        mN, SN, SN_inv = estimate_with(gamma, beta_, data_x, data_y)
        w = pyro.sample("w", pyro.distributions.MultivariateNormal(mN, precision_matrix=SN_inv)).to(data_x.device)

        return {"w": w, "y_noise": y_noise}

    if fit_intercept:
        X = np.concatenate([X, np.ones([len(X), 1])], axis=1)
        # some non-informative prior on intercept
        eps = np.insert(eps, len(eps), 1e-2)

    X_mu = np.expand_dims(X.mean(axis=0), axis=0)
    X = (X - X_mu)  # / np.expand_dims(X.std(axis=0), axis=0)
    X, y, eps = torch.tensor(X).type(torch.float32), torch.tensor(y).type(torch.float32), torch.tensor(eps).type(
        torch.float32)

    X, y, eps = X.to(device), y.to(device), eps.to(device)
    num_dim = X.shape[-1]
    if len(X) > 5000:
        select_idxs = np.random.choice(len(X), 2 * num_dim)
        X_, y_ = X[select_idxs], y[select_idxs]
    else:
        X_, y_ = X, y

    if (beta == None) or (gamma == None):
        pyro.clear_param_store()
        adam = pyro.optim.Adam({"lr": 1e-1})
        elbo = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(simple_model, custom_guide, adam, elbo)

        losses = []
        for step in range(100):
            loss = svi.step(X_, y_)
            losses.append(loss)
            if step % 100 == 0:
                print(f"step: {step}")
                print("Elbo loss: {}".format(loss))
                print("Optimal gamma:", pyro.param("gamma").detach().cpu().numpy(), " beta: ",
                      pyro.param("beta").detach().cpu().numpy())

        gamma, beta = float(pyro.param("gamma").detach().cpu()), float(pyro.param("beta").detach().cpu())
    # gamma, beta = 0, 1
    mN, SN, _ = estimate_with(gamma, beta, X, y)
    print("Optimal gamma:", gamma, "beta:", beta)

    mean, sigma = mN.detach().cpu().numpy(), np.sqrt(torch.diag(SN).detach().cpu().numpy())
    X, y = X.cpu().numpy(), y.cpu().numpy()

    if fit_intercept:
        mean, sigma = mean[:-1], sigma[:-1]
    return mean, sigma, beta, gamma
