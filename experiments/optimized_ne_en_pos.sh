NAME=$(basename "$0")

# Step 1: Preprocess data
python preprocess.py \
    --src ne \
    --tgt en \
    --part-of-speech

# Step 2: Train
bash ./train.sh ne en data

# Step 3: Evaluate
if [ ! -d "experiments/results" ]; then
    mkdir -p experiments/results
fi
bash ./evaluate.sh $NAME
