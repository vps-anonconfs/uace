import captum
import torch
from torch.utils.data import TensorDataset

import numpy as np
import torch.utils.data as data_utils

# hehe, one of those rare cases when the name is expanded
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"


class TCAVWrapper:
    """
    Failed attempt at trying to use Captum library implementation
    Captum has the worst documentation and does nothing by default not even p-score based concept filtering.
    Besides, it has such a weird design of interfaces that even controlling the device of computation 
    is not straight-forward. 
    """
    @staticmethod
    def compute(context, layer_name):
        tcav_scorer = captum.concept.TCAV(context.model.to(device), [layer_name])
        bs = 64
        if not context.concept_dataset:
            dl = data_utils.DataLoader(context.dataset, batch_size=bs, shuffle=True)
        else:
            dl = data_utils.DataLoader(context.concept_dataset, batch_size=bs, shuffle=True)

        all_x, all_g, all_y = [], [], []
        for batch_x, batch_y, batch_g in dl:
            all_x.append(batch_x.cpu())
            all_g.append(batch_g.cpu())
        all_x = torch.cat(all_x, dim=0)
        all_g = torch.cat(all_g, dim=0)

        num_concepts = all_g.shape[-1]
        concepts = []
        for ci in range(num_concepts):
            pos_idxs = np.where(all_g[:, ci] == 1)[0]
            if len(pos_idxs) < 5:
                continue

            dat = TensorDataset(all_x[pos_idxs].to(device))
            print(f"Num examples {len(dat)} for concept: {ci}")
            
            dl = data_utils.DataLoader(dat, batch_size=bs)
            this_concept = captum.concept.Concept(id=ci, name=context.concepts[ci], data_iter=dl)
            concepts.append(this_concept)
            
        print(f"Total concepts {len(concepts)}")
        concepts = concepts[:20]
        
        if context.concept_dataset:
            all_x = []
            dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=True)
            for batch_x, batch_y, batch_g in dl:
                all_x.append(batch_x)
            all_x = torch.cat(all_x, dim=0).to(device)
        scores = tcav_scorer.interpret(all_x, [concepts], target=0, processes=4)
        print(scores)
        return scores


class Squeezer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view([len(x), -1])
    

def split(model, layer_index):
    lst = list(model.children())
    m1 = torch.nn.Sequential(*lst[:layer_index])
    m2 = torch.nn.Sequential(*lst[layer_index:-1], Squeezer(), lst[-1])
    return m1, m2
    
class TCAV:
    @staticmethod
    def compute(context, layer_index):
        """
        Use g vector of the dataset to learn concept vectors and compute CAV scores
        1. Estimate concept vectors using the dataset
        2. Compute CAV scores per example and aggregate.
        :return: importance score per concept per class
        """
        reprs_layer, prediction_layer = split(context.model, layer_index)
        reprs_layer.eval()
        rng = np.random.default_rng(context.random_state)
        
        if not context.concept_dataset:
            dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)
        else:
            dl = data_utils.DataLoader(context.concept_dataset, batch_size=32, shuffle=False)
        all_reprs, all_g, all_y = [], [], []
        for batch_x, batch_y, batch_g in dl:
            all_reprs.append(reprs_layer(batch_x.to(device)).detach().cpu().numpy())
            all_g.append(batch_g.cpu().numpy())
            all_y.append(batch_y.cpu().numpy())
        all_reprs = np.concatenate(all_reprs)
        all_g = np.concatenate(all_g)
        all_y = np.concatenate(all_y)
        original_shape = all_reprs.shape
        all_reprs = all_reprs.reshape([len(all_reprs), -1])

        # get concept vectors and filter them based using num_sample MC
        num_samples = 10 # num_samples suggested in TCAV paper is way higher: 512
        sampled_cvs = [[] for _ in range(num_samples)]
        num_concepts = all_g.shape[-1]
        rng = np.random.default_rng(context.random_state)
        idxs = rng.permuted(np.arange(len(all_reprs)))
        train_ln = len(idxs) // 2
        train_idxs, eval_idxs = idxs[:train_ln], idxs[train_ln:]
        support_concepts = np.ones([num_concepts], np.int8)
        _num = 0
        for ci in range(num_concepts):
            # N x repr_size, N
            X, y = all_reprs[train_idxs], all_g[train_idxs, ci]
            pos_idxs, neg_idxs = np.where(y==1)[0], np.where(y==0)[0]
            X = np.concatenate([X[pos_idxs[:100]], X[neg_idxs[:100]]], axis=0)
            y = np.concatenate([y[pos_idxs[:100]], y[neg_idxs[:100]]], axis=0)

            idxs0, idxs1 = np.where(y == 0)[0], np.where(y == 1)[0]
            print(f"Num negative: {len(idxs0)} num positive: {len(idxs1)}")
            # no positive examples of this concept found, just set to random concept vector
            if (len(idxs1) < 5):
                support_concepts[ci] = 0
                for _ in range(num_samples):
                    sampled_cvs[_].append(rng.normal(size=[X.shape[-1]]))
                continue
            _num += 1

            for si in range(num_samples):
                _clf = LogisticRegression(penalty=None, fit_intercept=False, random_state=si)
                # subsetting for randomness: removed it because subsetting is sometimes taking away one of the classes.
                _clf.fit(X, y)
                sampled_cvs[si].append(_clf.coef_.squeeze())
            acc = _clf.score(X, y)
            print(f"Concept: {ci} -- acc: {acc}")

        for si in range(num_samples):
            # num_concepts x repr_dim
            sampled_cvs[si] = np.stack(sampled_cvs[si], axis=0)
            # sampled_cvs[si] /= np.expand_dims(np.linalg.norm(sampled_cvs[si], axis=0), axis=0)
        # step 2: compute cavs and aggregate
        # simple dot product if we are only dealing with last layer
        # final_layer: num_classes x repr_layer_size
        # final_layer_w = context.final_layer.weight.detach().numpy()
        # tcav_scores = np.matmul(final_layer_w, cvs)

        if context.concept_dataset:
            dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)
            all_reprs, all_g, all_y = [], [], []
            for batch_x, batch_y, batch_g in dl:
                all_reprs.append(context.reprs_layer(batch_x.to(device)).detach().cpu().numpy())
                all_g.append(batch_g.cpu().numpy())
                all_y.append(batch_y.cpu().numpy())
            all_reprs = np.concatenate(all_reprs)
            all_g = np.concatenate(all_g)
            all_y = np.concatenate(all_y)

            idxs = rng.permuted(np.arange(len(all_reprs)))
            train_ln = len(idxs) // 2
            train_idxs, eval_idxs = idxs[:train_ln], idxs[train_ln:]

        # compute number of examples for which the concept is important
        wts = []
        if context.labels_of_interest:
            labels = context.labels_of_interest
        else:
            labels = np.arange(np.max(all_y) + 1)

        for y in labels:
            idxs_for_label = np.where(all_y==y)[0]
            _reprs = all_reprs[idxs_for_label].reshape([len(idxs_for_label)] + list(original_shape[1:]))
            dl = data_utils.DataLoader(TensorDataset(torch.from_numpy(_reprs)), batch_size=32, shuffle=False)
            sampled_as = [[] for _ in range(num_samples)]
            for batch_reprs in dl: 
                batch_reprs = batch_reprs[0]
                batch_reprs.requires_grad = True
                batch_logits = prediction_layer(batch_reprs.to(device))
                # batch_size x repr_size
                grad_for_this_label = torch.autograd.grad(batch_logits[:, y].sum(), batch_reprs)[0].detach().cpu().numpy()
                grad_for_this_label = grad_for_this_label.reshape([len(batch_reprs), -1])
                for si in range(num_samples):
                    # batch_size x num_concepts
                    _a = np.matmul(grad_for_this_label, sampled_cvs[si].T)
                    sampled_as[si].append(_a)
            sampled_scores = []
            for si in range(num_samples):
                _a_matrix = np.concatenate(sampled_as[si], axis=0)
                # num_concepts x 1
                _s = np.where(_a_matrix>0, np.ones_like(_a_matrix), np.zeros_like(_a_matrix)).mean(axis=0)
                sampled_scores.append(_s)
            # num_samples x num_concepts
            sampled_scores = np.stack(sampled_scores, axis=0)
            tcav_mean, tcav_std = sampled_scores.mean(axis=0), sampled_scores.std(axis=0)
            # print(tcav_mean[:20], sampled_scores[0][:20])
            # two-tailed significance testing for deviation from 0.5
            val = np.abs(tcav_mean - 0.5)/np.sqrt(tcav_std**2/num_samples)
            # size of num_concepts 
            p_score = 2*student_t.sf(val, num_samples-1)
            assert len(p_score) == len(tcav_mean), f"Expected length of pscores is {len(tcav_mean)}, but found {len(p_score)}"
            scores = np.where((p_score <= 0.01) & (support_concepts == 1), tcav_mean, np.zeros_like(tcav_mean))
            wts.append(tcav_mean)
            
        wts = np.stack(wts, axis=0)
        return wts
    
    
class TCAV_v2:
    """
    This is (somewhat) of an unfaithful implementation of TCAV where a pseudo TCAV score is computed 
    through \sum S_{ckl}=p(c|y=k) using concept vectors estimated on the last layer. 

    On the other hand, TCAV estimates scores using a causal dynamic p(y=k|do(c)) by locally 
    perturbing the concept. Despite the difference, believe they must be similar, as was also 
    observed in this paper: https://openreview.net/forum?id=J0qgRZQJYX
    """
    @staticmethod
    def compute(context):
        """
        Use g vector of the dataset to learn concept vectors and compute CAV scores
        1. Estimate concept vectors using the dataset
        2. Compute CAV scores per example and aggregate.
        :param model:
        :param dataset:
        :param concepts:
        :return: importance score per concept per class
        """
        if not context.concept_dataset:
            dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)
        else:
            dl = data_utils.DataLoader(context.concept_dataset, batch_size=32, shuffle=False)
        all_reprs, all_g, all_y = [], [], []
        for batch_x, batch_y, batch_g in dl:
            all_reprs.append(context.reprs_layer(batch_x).detach().cpu().numpy())
            all_g.append(batch_g.cpu().numpy())
            all_y.append(batch_y.cpu().numpy())
        all_reprs = np.concatenate(all_reprs)
        all_g = np.concatenate(all_g)
        all_y = np.concatenate(all_y)

        # get concept vectors
        cvs = []
        num_concepts = all_g.shape[-1]
        rng = np.random.default_rng(context.random_state)
        idxs = rng.permuted(np.arange(len(all_reprs)))
        train_ln = len(idxs) // 2
        train_idxs, eval_idxs = idxs[:train_ln], idxs[train_ln:]
        for ci in range(num_concepts):
            _clf = LogisticRegression(penalty=None, fit_intercept=False, random_state=context.random_state)
            # N x repr_size, N
            X, y = all_reprs[train_idxs], all_g[train_idxs, ci]

            idxs0, idxs1 = np.where(y == 0)[0], np.where(y == 1)[0]
            print(f"Num negative: {len(idxs0)} num positive: {len(idxs1)}")
            # no positive examples of this concept found, just set to random concept vector
            if len(idxs1) == 0:
                cvs.append(rng.normal(size=[X.shape[-1]]))
                continue

            _clf.fit(X, y)
            cvs.append(_clf.coef_.squeeze())
            acc = _clf.score(X, y)
            print(f"Concept: {ci} -- acc: {acc}")

        cvs = np.stack(cvs, axis=-1)
        cvs /= np.expand_dims(np.linalg.norm(cvs, axis=0), axis=0)
        # print("dps:", np.matmul(cvs.T, cvs))
        # step 2: compute cavs and aggregate
        # simple dot product if we are only dealing with last layer
        # final_layer: num_classes x repr_layer_size
        # final_layer_w = context.final_layer.weight.detach().numpy()
        # tcav_scores = np.matmul(final_layer_w, cvs)

        if context.concept_dataset:
            dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)
            all_reprs, all_g, all_y = [], [], []
            for batch_x, batch_y, batch_g in dl:
                all_reprs.append(context.reprs_layer(batch_x).detach().cpu().numpy())
                all_g.append(batch_g.cpu().numpy())
                all_y.append(batch_y.cpu().numpy())
            all_reprs = np.concatenate(all_reprs)
            all_g = np.concatenate(all_g)
            all_y = np.concatenate(all_y)

            idxs = rng.permuted(np.arange(len(all_reprs)))
            train_ln = len(idxs) // 2
            train_idxs, eval_idxs = idxs[:train_ln], idxs[train_ln:]

        # compute number of examples for which the concept is important
        wts = []
        if context.labels_of_interest:
            labels = context.labels_of_interest
        else:
            labels = np.arange(np.max(all_y) + 1)

        for y in labels:
            this_y_idxs = np.where(all_y[eval_idxs] == y)[0]
            _idxs = eval_idxs[this_y_idxs]
            concept_scores = np.matmul(all_reprs[_idxs], cvs)
            wts.append(np.mean(concept_scores > 0, axis=0))
        wts = np.stack(wts, axis=0)

        return wts