from typing import List

import torch
from clip import clip

from explanations.basic import OracleRegressionFit, OracleBayesFit
from explanations.ocbm import OCBM
from explanations.tcav import TCAV
from explanations.uace import UACE, NOISE_MODELLING
from explanations.ycbm import YCBM

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


class ConceptExplainer:
    def __init__(self, prediction_model, reprs_layer, final_layer, probe_dataset, num_classes,
                 concepts=[], random_state=0, custom_clip_model=None, labels_of_interest=None,
                 concept_dataset=None):
        """
        ToDO: Get rid of final_layer param, not used anywhere. 
        :param prediction_model: takes input and returns logits
        :param reprs_layer: takes input and returns representations
        :param probe_dataset: dataset of examples for explanation computation 
        :param concepts: list of text literals identifying the concept, these are used to feed clip
        :param random_state:
        :param custom_clip_model: use this model instead of the default (clip), may come handy when clip representations are cached, should take batch of instances from probe_dataset as input and return a torch tensor of size batch_size x num_concepts
        :param labels_of_interest: A list of labels for which we wish to see explanation. Useful for computational reasons
        :param concept_dataset: If set, TCAV uses this instead of probe_dataset for learning concepts
        """
        self.dataset = probe_dataset
        self.model, self.reprs_layer = prediction_model, reprs_layer
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        if hasattr(self.reprs_layer, 'to'):
            self.reprs_layer = self.reprs_layer.to(device)
        self.final_layer = final_layer
        self.concepts = concepts
        self.num_classes = num_classes

        self.custom_clip_model = None
        if not custom_clip_model:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        else:
            self.custom_clip_model = custom_clip_model
        self.random_state = random_state
        torch.random.manual_seed(self.random_state)

        self.labels_of_interest = labels_of_interest
        self.concept_dataset = concept_dataset

    def set_concept_names(self, concept_names: List):
        self.concepts = concept_names

    def simple_fit(self, lasso_alpha=1e-2):
        return OracleRegressionFit.compute(self, lasso_alpha)

    def bayesian_fit(self):
        return OracleBayesFit.compute(self)

    def tcav(self, layer_index):
        """
        :param layer_name: is the name of the layer that is used for computing the explanation. 
                           The layer_name should be one found in model.__dict__ and 
                           getattr(model, layer_name) should give a valid module. 
        """
        return TCAV.compute(self, layer_index)

    # TODO: cleanup the args
    def oikarinen_cbm(self, return_acc=False, fit_intercept=True, return_concept_reprs=False, lasso_C=1):
        return OCBM.compute(self, return_acc, fit_intercept, return_concept_reprs, lasso_C=lasso_C)

    def ycbm(self, return_acc=False, fit_intercept=True, lasso_alpha=1e-2):
        return YCBM.compute(self, return_acc, fit_intercept, lasso_alpha)

    def uace(self, mode=NOISE_MODELLING.FULL, return_acc=False, kappa=0.005):
        return UACE.compute(self, mode, return_acc, kappa)