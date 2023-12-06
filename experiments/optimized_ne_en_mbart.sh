NAME=$(basename "$0")

# Step 1: Preprocess data
python preprocess.py \
    --src ne \
    --tgt en \
    --mBART

# Step 2: Pretrain
bash ./train.sh ne en data/pretrain

# Step 3: Finetune
bash ./finetune.sh ne en data

# Step 4: Evaluate
if [ ! -d "experiments/results" ]; then
    mkdir -p experiments/results
fi
bash ./evaluate.sh $NAME checkpoint_last
