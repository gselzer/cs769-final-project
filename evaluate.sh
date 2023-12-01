SCRIPT=$1
NAME=${SCRIPT%.*}
RESULT_DIR="experiments/results/$NAME"

# Build the results
rm -rf $RESULT_DIR
mkdir -p $RESULT_DIR
cp checkpoints/checkpoint_best.pt $RESULT_DIR/checkpoint.pt
if [ ! -d "data/pretrain" ]; then
	cp -r data/pretrain $RESULT_DIR/pretrain_data
fi
cp -r data $RESULT_DIR/data

# Test the model
fairseq-generate data-bin \
    --path $RESULT_DIR/checkpoint.pt \
    --max-tokens 4096 --beam 5 \
    --results-path $RESULT_DIR \
    --sacrebleu

exit 0

