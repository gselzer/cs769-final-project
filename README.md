# CS769 Final Project - Gabriel Selzer, Luke Neuendorf, Zach Potter

Many **Low-resource** machine translation problems cannot afford the hardware necessary for solutions involving complex language models. In such cases, improving accuracy must be accomplished using other techniques. Our final project seeks to understand how well state-of-the-art techniques for improving accuracy in low-resource situations fare with a *small* language model, across varying degrees of language similarity and varying amounts of technique composition.

## Proposal Information
For our proposal, we implement Joint Dropout ([paper](https://arxiv.org/pdf/2307.12835v1.pdf))

### Setup
Build and activate the conda environment:
```
conda env create -f environment.yml
conda activate cs769
```
Run the training pipeline:
```
python preprocess.py && ./preprocess.sh && ./train.sh
```
