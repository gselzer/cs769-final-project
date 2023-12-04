NAME=$(basename "$0")

# Step 1: Preprocess data
python preprocess.py \
    --src ne \
    --tgt en \
    --joint-dropout \
#    --gen-jd-data # uncomment this to run JD data generation, however, we had problems running it
#    on Windows but it works on MacOS

# # Step 2: Pretrain
# bash ./train.sh ne en data/pretrain


# Step 3: Finetune
bash ./finetune.sh ne en data

# Step 4: Evaluate
if [ ! -d "experiments/results" ]; then
    mkdir -p experiments/results
fi
bash ./evaluate.sh $NAME
