NAME=$(basename "$0")
SRC=ne
TGT=en

# Step 1: Preprocess data
python preprocess.py \
    --src $SRC \
    --tgt $TGT \

# Step 2: Finetune
bash ./finetune.sh $SRC $TGT

# Step 3: Evaluate
if [ ! -d "experiments/results" ]; then
    mkdir -p experiments/results
fi
bash ./evaluate.sh $NAME
