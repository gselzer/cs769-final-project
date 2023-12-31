rm -rf data-bin

# Step 1: Binarize data
SRC=$1
TGT=$2
TEXT=$3
fairseq-preprocess --source-lang $SRC --target-lang $TGT \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin \
    --srcdict data/join-dict.txt \
    --tgtdict data/join-dict.txt \
    --workers 20

# Step 2: Train using fairseq
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --source-lang $SRC \
    --target-lang $TGT \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.5 \
    --max-tokens 4096 \
    --scoring sacrebleu \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --encoder-layers 5 --decoder-layers 5\
    --encoder-attention-heads 2 --decoder-attention-heads 2\
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --encoder-layerdrop 0.0 --decoder-layerdrop 0.2 \
    --activation-dropout 0.3 \
    --max-epoch 300 \
    --save-interval 10 \
    --validate-interval 10 \
