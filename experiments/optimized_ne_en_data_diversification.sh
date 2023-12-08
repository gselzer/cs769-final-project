#!/bin/bash

# Run data diversificaiton which preprocesses the data and trains the models
# NOTE: to run using GPU, remove the flag "--use_cpu" to the python command below. However, we had CUDA
# memory errors when running with gpu
python dd_run.py --k 3 --N 1 --n_epoch 60 --arch_fwd "transformer_iwslt_de_en" --arch_bkwd \
    "transformer_wmt_en_de" --src_lang "ne" --trg_lang "en" 
# Create results dir if it doesn't exist
DIR="experiments/results/optimized_ne_en_data_diversification"
if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR"
fi

# Copy model file to the new results path
cp models/checkpoint_best_fwd.pt "$DIR/checkpoint.pt"

# Copy data to the new results path
cp -r data/ "$DIR/"

# Copy test output to the new results path
cp results/generate-test.txt "$DIR/generate-test.txt"

# Copy intermediate output to the new results path
cp -r intermediate-model-outputs "$DIR"
