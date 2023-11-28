SCRIPT=$1
NAME=${SCRIPT%.*}
RESULT_DIR="experiments/results/$NAME"

rm -rf $RESULT_DIR
mkdir -p $RESULT_DIR
cp checkpoints/checkpoint_best.pt $RESULT_DIR/checkpoint.pt

fairseq-generate data-bin \
    --path $RESULT_DIR/checkpoint.pt \
    --max-tokens 4096 --beam 5 \
    --results-path $RESULT_DIR \
    --sacrebleu

exit 0

