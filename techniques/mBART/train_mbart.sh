NAME=$(basename "$0")
SRC=de
TGT=en

# Step 1: Preprocess data
python preprocess.py \
    --src $SRC \
    --tgt $TGT \
    --mBART

# Step 2: Pretrain
bash ./train.sh $SRC $TGT data/pretrain

# Step 3: Finetune
bash ./finetune.sh $SRC $TGT data

# Step 4: Evaluate
if [ ! -d "experiments/results" ]; then
    mkdir -p experiments/results
fi
bash ./evaluate.sh $NAME
