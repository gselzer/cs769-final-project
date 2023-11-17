import os
import re
import random
from techniques.interleaving.linguistic import tag
from techniques.mRASP.mrasp import mRASP
import utils

from mBART import mbart

# Configuration Constants
DATA_DIR = "data/"
TEMP_DIR = "temp/"
SRC_LANG = "de"
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

utils.develop_data(TEMP_DIR, SRC_LANG, TGT_LANG)

utils.preprocess_data(TEMP_DIR, SRC_LANG, TGT_LANG)

# # Create RAS pretraining data
# mRASP(
#     [f"{TEMP_DIR}tmp.train.{SRC_LANG}", f"{TEMP_DIR}tmp.valid.{SRC_LANG}", f"{TEMP_DIR}tmp.test.{SRC_LANG}"],
#     [f"{TEMP_DIR}tmp.train.{TGT_LANG}", f"{TEMP_DIR}tmp.valid.{TGT_LANG}", f"{TEMP_DIR}tmp.test.{TGT_LANG}"],
#     f"{TEMP_DIR}pretrain"
# )
# POS tag data
# for s in ["train", "test", "valid"]:
#     # Only add POS for source data
#     for l in [SRC_LANG]:
#         tag(f"{TEMP_DIR}tmp.{s}.{l}", l)
mbart(
    [f"{TEMP_DIR}tmp.train.en", f"{TEMP_DIR}tmp.train.de"], 
    f"{TEMP_DIR}pretrain")


# Learn BPE
num_bpe_tokens: int = 10000
os.system(f"subword-nmt learn-joint-bpe-and-vocab --input {TEMP_DIR}tmp.train.{TGT_LANG} {TEMP_DIR}tmp.train.{SRC_LANG} \
            -s {num_bpe_tokens} -o {TEMP_DIR}code.txt --write-vocabulary {TEMP_DIR}vocab.{TGT_LANG} {TEMP_DIR}vocab.{SRC_LANG}")


# # Joint Dropout
# wa = utils.WordAligner()
# wa.word_alignments(source_file=f"{TEMP_DIR}tmp.train.de",
#                     target_file=f"{TEMP_DIR}tmp.train.en",
#                     output_file=f'{TEMP_DIR}eflomal.de.en',
#                     model = '3')
# JDR = utils.JointDropout(debug=False)
# JDR.joint_dropout(f"{TEMP_DIR}tmp.train.de", f"{TEMP_DIR}tmp.train.en", f'{TEMP_DIR}eflomal.de.en',
#                   output_dir=f'{TEMP_DIR}', src_suffix='de',trg_suffix='en')

# # Concatenate JDR output with the tmp.train.en/tmp.train.de files.
# with open(f'{TEMP_DIR}jdr.src.de', 'r') as source_file:
#     data_to_append = source_file.read()

# with open(f'{TEMP_DIR}tmp.train.de', 'a') as target_file:
#     target_file.write(data_to_append)

# with open(f'{TEMP_DIR}jdr.trg.en', 'r') as source_file:
#     data_to_append = source_file.read()

# with open(f'{TEMP_DIR}tmp.train.en', 'a') as target_file:
#     target_file.write(data_to_append)

# Apply BPE
glossary = [
    # Joint Dropout
    "<S_\d+>",
    "<T_\d+>",
    # Penn Treebank POS tags
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    # MBART
    "<mask>",
]
glossary_str = " ".join([f"'{d}'" for d in glossary])
for s in ["train", "test", "valid"]:
    for l in [SRC_LANG, TGT_LANG]:
        os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.{l} < \
                    {TEMP_DIR}tmp.{s}.{l} > {DATA_DIR}{s}.{l} --glossaries {glossary_str}")

# Apply BPE to pretrained data
os.makedirs(f"{DATA_DIR}pretrain")
for s in ["train", "test", "valid"]:
    for l in [SRC_LANG, TGT_LANG]:
        os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.en < \
                    {TEMP_DIR}pretrain/{s}.{l} > {DATA_DIR}pretrain/{s}.{l} --glossaries {glossary_str}")

os.system(f"cp {TEMP_DIR}code.txt {DATA_DIR}code.txt")

# Delete Temp Files
if DELETE_TEMP_DATA:
    os.system("rm -rf temp")
