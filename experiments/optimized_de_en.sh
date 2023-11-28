
# Step 1: Preprocess data
python preprocess.py \
    --src de \
    --tgt en \

# Step 2: Finetune
bash ./finetune.sh

# Step 3: Evaluate
bash ./evaluate.sh > results/optimized_de_en.sh