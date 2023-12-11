# CS769 Final Project - Gabriel Selzer, Luke Neuendorf, Zach Potter

This final project was motivated by the observation that advancements in all machine learning, including machine translation, trend towards using more data and larger models trained on faster hardware. As a result, problems with fewer resources and researchers without access to powerful hardware are left unable to utilize these new advancements. This project seeks to determine whether modern techniques, trained with either a large model, or large datasets, or both, can be transferred to a smaller model and a smaller dataset to a similar effect as they showed in their original results.

## Experiments

This project reimplemented 5 different techniques written within the last 4 years:
* [mBART - Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210)
* [mRASP - Pre-training Multilingual Neural Machine Translation by Leveraging Alignment Information](https://arxiv.org/abs/2010.03142)
* [Joint Dropout - Improving Generalizability in Low-Resource Neural Machine Translation through Phrase Pair Variables](https://arxiv.org/abs/2307.12835)
* [Data Diversification - A Simple Strategy For Neural Machine Translation](https://arxiv.org/abs/1911.01986)
* [Part-of-Speech Tagging - Understanding the effects of word-level linguistic annotations in under-resourced neural machine translation](https://aclanthology.org/2020.coling-main.349/)

The techniques described in each paper were transferred to a generic Transformer model, implemented within the [Fairseq](https://aclanthology.org/N19-4009/) framework - this framework was chosen for its accessibility and its performance.

For each derived model, we trained translation models on two different datasets:
1. The German-English parallel dataset utilized in the [WMT 2014 Shared Task](https://statmt.org/wmt14/translation-task.html#Download)
2. The Nepali-English parallel dataset utilized in the [FLORES Evaluation Benchmark](https://arxiv.org/abs/2106.03193)

These particular datasets were chosen to represent linguistically similar (German-English) and different (Nepali-English) languages. Each was sampled to 10 thousand and 20 thousand random samples, respectively, to enable rapid training and development by a researcher on consumer hardware in a couple of hours.

## Reproducing Results

To reproduce the results of our experiments, you must first install the necessary environment components, through the provided `environment.yml`:

```bash
conda env create -f environment.yml
```

The `experiments` directory contains an executable script for each combination of parallel dataset and implemented technique (including the baseline), for 12 total scripts. Each will locally cache the training data for the specified language pair, prepare the data for training using the specified technique, and then train a model for a pre-specified 150 epochs. Finally, each script will deposit the results of training in an `experiments/results` subfolder for further analysis.

For example, you can run the following to preprocess our German-English translation task dataset, train a baseline model, and generate an output sacreBLEU score:
```bash
bash ./experiments/optimized_de_en.sh
```
That final score can be printed using the following command:
```bash
tail -n 1 ./experiments/results/optimized_de_en/generate-test.txt
```

Our analysis was performed using the [`fairseq-generate`](https://fairseq.readthedocs.io/en/latest/command_line_tools.html) shipped with the fairseq library, and with the [compare-mt](https://arxiv.org/abs/1903.07926https://github.com/neulab/compare-mt) project. Our compare-mt results can be approximated (considering training randomization) using the `compare-models.py` script:

```bash
python compare_models.py --m1 experiments/results/optimized_de_en --m2 experiments/results/optimized_de_en_mrasp
```

Add "--st" to include significance testing along with the normal compare-mt results:
```bash
python compare_models.py --m1 experiments/results/optimized_de_en --m2 experiments/results/optimized_de_en_mrasp --st
```

# Proposal Submission

Many **Low-resource** machine translation problems cannot afford the hardware necessary for solutions involving complex language models. In such cases, improving accuracy must be accomplished using other techniques. Our final project seeks to understand how well state-of-the-art techniques for improving accuracy in low-resource situations fare with a *small* language model, across varying degrees of language similarity and varying amounts of technique composition.

## Proposal Information (Functionality deprecated)
For our proposal, we implement Joint Dropout ([paper](https://arxiv.org/pdf/2307.12835v1.pdf))

### Setup
Build and activate the conda environment:
```
conda env create -f environment.yml
conda activate cs769
```
Run the training pipeline:
```
python preprocess.py && ./binarize.sh && ./train.sh
```
