import torch
import torch.utils.data as data_utils
import numpy as np
from sklearn.linear_model import Lasso, BayesianRidge

device = "cuda" if torch.cuda.is_available() else "cpu"

class OracleRegressionFit:
    """
    Fits a linear regression function on attributes in order to regress model predictions

    :param model: model that is to be explained
    :param dataset: probe dataset for computing explanation
    :param concepts: list of all concepts
    :return: relative importance of concepts
    """
    @staticmethod
    def compute(context, lasso_alpha):
        dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)
        all_logits = []
        all_attrs = []
        for batch_x, batch_y, batch_g in dl:
            batch_x = batch_x.to(device)
            logits = context.model(batch_x)
            probs = logits.softmax(dim=-1)
            all_logits.append(logits.detach().cpu().numpy())
            all_attrs.append(batch_g.cpu().numpy())

        X, y = np.concatenate(all_attrs), np.concatenate(all_logits, axis=0)
        num_classes = y.shape[-1]
        all_coefs = []
        for ci in range(num_classes):
            fitter = Lasso(alpha=lasso_alpha, fit_intercept=True, random_state=context.random_state)
            fitter.fit(X, y[:, ci])
            all_coefs.append(fitter.coef_)

        all_coefs = np.array(all_coefs)
        all_attrs = np.concatenate(all_attrs, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        if context.labels_of_interest:
            all_coefs = np.array([all_coefs[_y] for _y in context.labels_of_interest])
            labels = np.array(context.labels_of_interest)
        else:
            labels = np.arange(all_logits.shape[1])

        true_preds = np.argmax(all_logits[:, labels], axis=-1)
        approx_preds = np.argmax(np.matmul(all_attrs, np.transpose(all_coefs)), axis=-1)
        print("Acc", np.equal(true_preds, approx_preds).mean())
        return all_coefs


class OracleBayesFit:
    def compute(context):
        """
        :param model: model that is to be explained
        :param dataset: probe dataset for computing explanation
        :param concepts: list of all concepts
        :return: relative importance of concepts
        """
        dl = data_utils.DataLoader(context.dataset, batch_size=32, shuffle=False)
        all_logits = []
        all_attrs = []
        for batch_x, batch_y, batch_g in dl:
            batch_x = batch_x.to(device)
            logits = context.model(batch_x)
            probs = logits.softmax(dim=-1)
            all_logits.append(logits.detach().cpu().numpy())
            all_attrs.append(batch_g.cpu().numpy())

        X, y = np.concatenate(all_attrs), np.concatenate(all_logits, axis=0)
        num_classes = y.shape[-1]
        all_coefs = []
        for ci in range(num_classes):
            fitter = BayesianRidge(fit_intercept=True)
            fitter.fit(X, y[:, ci])
            # print("Sigma", fitter.sigma_)
            all_coefs.append(fitter.coef_)

        # any sigma is fine. Assuming they are all the same.
        return np.array(all_coefs), fitter.sigma_
