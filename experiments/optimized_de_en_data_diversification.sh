#!/bin/bash

# Change directory to two levels up where dd_train.py is located
cd ../

# Run data diversificaiton which preprocesses the data and trains the models
python dd_run.py --k 3 --N 1 --n_epoch 60 --use_gpu --arch_fwd "transformer_iwslt_de_en" --arch_bkwd \
    "transformer_wmt_en_de" --src_lang "de" --trg_lang "en"


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
