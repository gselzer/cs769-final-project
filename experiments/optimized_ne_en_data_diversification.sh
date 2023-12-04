#!/bin/bash

# Change directory to two levels up where dd_train.py is located
cd ../

# Run data diversificaiton which preprocesses the data and trains the models
# NOTE: to run using GPU, remove the flag "--use_cpu" to the python command below. However, we had CUDA
# memory errors when running with gpu
python dd_run.py --k 3 --N 1 --n_epoch 60 --arch_fwd "transformer_iwslt_de_en" --arch_bkwd \
    "transformer_wmt_en_de" --src_lang "ne" --trg_lang "en" --use_cpu


# Delete added directories, besided data-bin (contains the binarized partially synthetic dataset) and models
#directories=("translations" "temp" "data")

#for dir in "${directories[@]}"; do
#    if [ -d "$dir" ]; then
#        echo "Deleting directory: $dir"
#        rm -rf "$dir"
#    else
#        echo "Directory does not exist: $dir"
#    fi
#done
