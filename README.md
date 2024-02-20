Contains various implementations of Concept-based explanations, and is the official implementation of 
Uncertainty-Aware Concept Explanations (U-ACE).

All plots generated using the ipynb notebook: `notebooks/plots.ipynb`

## Section 5 Experiments (STL with a tag)
Dataset: `mydatasets/simple_tag_dataset.py`   
Figure 2 (middle): `tag_expt.py` to generate explanations.   
Figure 2 (right): `tag_overcomplete_expt.py` to generate explanation with nuisance concepts  

## Section 6.1 Experimets (on Broden dataset with PASCAL and ADE20K)
Dataset: `mydatasets/broden_dataset.py`   
Features are generated and cached using `mydatasets/broden_features.py`   
`broden_expt.py` was used to generate explanations, which are used to generate Figure 3, Table 1, 2 by `notebooks/plots.ipynb`.   

## Section 6.2 Experiments (on Salient-Imagenet)
Dataset: `mydatasets/salient_imagenet.py`   
`simagenet_expts.py` caches the features and computes the explanation, which were then processed using `notebooks/simagenet.ipynb` to produce Table 3, 4, 7. 

## Uncertainty Evaluation of Appendix H
`eval_uncert.py`, `eval_uncert2.py` estimate the confidence intervals on concept activations using MC sampling, Distribution Fit (that is described in the paper: Appendix H.1) and ground-truth epistemic uncertainty. 

Table 9, Figures 7, 8, 9, 10 are generated using `notebooks/plots.ipynb`. 
