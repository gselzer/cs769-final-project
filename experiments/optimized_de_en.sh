NAME=$(basename "$0")

# Step 1: Preprocess data
python preprocess.py \
    --src de \
    --tgt en \

# Step 2: Finetune
bash ./train.sh de en data

# Step 3: Evaluate
if [ ! -d "experiments/results" ]; then
    mkdir -p experiments/results
fi
bash ./evaluate.sh $NAME
