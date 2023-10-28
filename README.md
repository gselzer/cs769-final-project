# Resources

## Joint Dropout ([paper](https://arxiv.org/pdf/2307.12835v1.pdf))

Note [this slightly older paper](https://aclanthology.org/2020.coling-main.304.pdf) from the same authors - their preprocessing pipeline seems similar ([code](https://github.com/aliaraabi/OptTransformer))

[Eflomal](https://github.com/robertostling/eflomal) - word alignment tool for generating symmetrized word alignments

[Moses](https://github.com/moses-smt/mosesdecoder), used for "punctuation normalization, tokenization, data cleaning, and true-casing" (but [this site](http://www.statmt.org/moses/))

[BPE Segmentation](https://github.com/rsennrich/subword-nmt)

[English-German TED training data](https://wit3.fbk.eu/2014-01)

[CoreNLP](https://stanfordnlp.github.io/CoreNLP/) ([paper](https://stanfordnlp.github.io/CoreNLP/)), used to create constituency trees for the training sentences.


# Setup
Build and activate the conda environment:
```
conda env create -f environment.yml
conda activate cs769
```
Run the training pipeline:
```
python preprocess.py && ./preprocess.sh && ./train.sh
```
