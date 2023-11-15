import os
import re
import random
import utils
from techniques.mRASP.mrasp import mRASP

# Configuration Constants
DATA_DIR = "data/"
TEMP_DIR = "temp/"
SRC_LANG = "ne"
TGT_LANG = "en"
DELETE_TEMP_DATA = True
NUM_THREADS = 1


# Delete data from old runs in "data" and "temp" directories
for dir_path in [TEMP_DIR, DATA_DIR]:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        os.system(f"rm -rf {dir_path}")

# Make temp and data directories
os.makedirs('data', exist_ok=True)
os.makedirs('temp', exist_ok=True)

utils.develop_ne_en_data(TEMP_DIR)

# Tokenize the data
os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l {TGT_LANG} \
            < {TEMP_DIR}tmp.{TGT_LANG} > {TEMP_DIR}tmp.tok.{TGT_LANG}")
os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l {SRC_LANG} \
            < {TEMP_DIR}tmp.{SRC_LANG} > {TEMP_DIR}tmp.tok.{SRC_LANG}")

# Clean the data
os.system(f"perl mosesdecoder/scripts/training/clean-corpus-n.perl {TEMP_DIR}tmp.tok {SRC_LANG} {TGT_LANG} {TEMP_DIR}tmp.clean 1 175")

# Truecase (lowercase) the data
os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.{SRC_LANG} > {TEMP_DIR}tmp.train.{SRC_LANG}")
os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.{TGT_LANG} > {TEMP_DIR}tmp.train.{TGT_LANG}")

# Sample the data
no_samples = 10000
with open(TEMP_DIR+f"tmp.train.{TGT_LANG}") as f:
    train_en = f.read().split("\n")
with open(TEMP_DIR+f"tmp.train.{SRC_LANG}") as f:
    train_de = f.read().split("\n")
samples = random.sample(range(len(train_en)), min(len(train_en), no_samples))
train_en = [train_en[i] for i in samples]
train_de = [train_de[i] for i in samples]
with open(TEMP_DIR+"tmp.train.{TGT_LANG}", "w") as f:
    f.write("\n".join(train_en))
with open(TEMP_DIR+"tmp.train.{SRC_LANG}", "w") as f:
    f.write("\n".join(train_de))


# Tokenize and clean test/validation data
for s in ["test", "valid"]:
    for l in [SRC_LANG, TGT_LANG]:
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l {l} \
                    < {TEMP_DIR}tmp.{s}.{l} > {TEMP_DIR}tmp.tok.{l}")
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.tok.{l} > \
                    {TEMP_DIR}tmp.{s}.{l}")

# Learn BPE
num_bpe_tokens: int = 10000
os.system(f"subword-nmt learn-joint-bpe-and-vocab --input {TEMP_DIR}tmp.train.{TGT_LANG} {TEMP_DIR}tmp.train.{SRC_LANG} \
            -s {num_bpe_tokens} -o {TEMP_DIR}code.txt --write-vocabulary {TEMP_DIR}vocab.{TGT_LANG} {TEMP_DIR}vocab.{SRC_LANG}")

# # RAS
# mRASP(f"{TEMP_DIR}tmp.train.de", f"{TEMP_DIR}tmp.train.{TGT_LANG}")

# # Joint Dropout
# wa = utils.WordAligner()
# wa.word_alignments(source_file=f"{TEMP_DIR}tmp.train.{SRC_LANG}",
#                     target_file=f"{TEMP_DIR}tmp.train.{TGT_LANG}",
#                     output_file=f'{TEMP_DIR}eflomal.{SRC_LANG}.{TGT_LANG}',
#                     model = '3')
# JDR = utils.JointDropout(debug=False)
# JDR.joint_dropout(f"{TEMP_DIR}tmp.train.{SRC_LANG}", f"{TEMP_DIR}tmp.train.{TGT_LANG}", f'{TEMP_DIR}eflomal.{SRC_LANG}.{TGT_LANG}',
#                   output_dir=f'{TEMP_DIR}', src_suffix=SRC_LANG,trg_suffix=TGT_LANG)

# # Concatenate JDR output with the tmp.train.en/tmp.train.de files.
# with open(f'{TEMP_DIR}jdr.src.{TGT_LANG}', 'r') as source_file:
#     data_to_append = source_file.read()

# with open(f'{TEMP_DIR}tmp.train.{TGT_LANG}', 'a') as target_file:
#     target_file.write(data_to_append)

# with open(f'{TEMP_DIR}jdr.trg.{TGT_LANG}', 'r') as source_file:
#     data_to_append = source_file.read()

# with open(f'{TEMP_DIR}tmp.train.{TGT_LANG}', 'a') as target_file:
#     target_file.write(data_to_append)

# Apply BPE
for s in ["train", "test", "valid"]:
    for l in [SRC_LANG, TGT_LANG]:
        os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.{l} < \
                    {TEMP_DIR}tmp.{s}.{l} > {DATA_DIR}{s}.{l} --glossaries '<S_\d+>' '<T_\d+>'")

# Delete Temp Files
if DELETE_TEMP_DATA:
    os.system("rm -rf temp")
