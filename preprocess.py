import os
import re
import random
from typing import List
from mosestokenizer import MosesPunctuationNormalizer, MosesTokenizer
from subword_nmt import learn_bpe, apply_bpe
from sample import read_source_and_target

def read_and_clean(infile: str, outfile: str):
    # This pattern will search for any tags
    delete_tags = ['<url>', '<talkid>', '<keywords>']
    clean_tags = ['<title>', "</title>", "<description>", "</description>"]

    # Read in the files
    with open(infile) as f:
        text_source = f.read().split('\n')
    
    cleaned = []
    for line in text_source:
        if any([tag in line for tag in delete_tags]):
            continue
        for tag in clean_tags:
            if tag in line:
                line = line.replace(tag, "")
        cleaned.append(line)

    # Write out the files
    with open(outfile, "w") as f:
        f.write("\n".join(cleaned))

def remove_if_exists(files: List[str]):
    for file in files:
        if os.path.exists(file):
            os.remove(file)
remove_if_exists(["train.de", "train.en", "valid.de", "valid.en", "test.de", "test.en", "tmp.valid.en", "tmp.valid.de", "tmp.test.de", "tmp.test.en"])


# Sample source and target
read_and_clean("de-en/train.tags.de-en.en", "tmp.en")
read_and_clean("de-en/train.tags.de-en.de", "tmp.de")

# Tokenize the data
os.system("perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l en < tmp.en > tmp.tok.en")
os.system("perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l de < tmp.de > tmp.tok.de")

# Clean the data
os.system("perl mosesdecoder/scripts/training/clean-corpus-n.perl tmp.tok de en tmp.clean 1 175")

# Truecase (lowercase) the data
os.system("perl mosesdecoder/scripts/tokenizer/lowercase.perl < tmp.clean.en > tmp.train.en")
os.system("perl mosesdecoder/scripts/tokenizer/lowercase.perl < tmp.clean.de > tmp.train.de")

# Sample the data
no_samples = 10000
with open("tmp.train.en") as f:
    train_en = f.read().split("\n")
with open("tmp.train.de") as f:
    train_de = f.read().split("\n")
samples = random.sample(range(len(train_en)), no_samples)
train_en = [train_en[i] for i in samples]
train_de = [train_de[i] for i in samples]
with open("tmp.train.en", "w") as f:
    f.write("\n".join(train_en))
with open("tmp.train.de", "w") as f:
    f.write("\n".join(train_de))

# Grab test/validation data
for file in ['IWSLT14.TED.tst2010', 'IWSLT14.TED.tst2011', 'IWSLT14.TED.tst2012']:
    for l in ['de', 'en']:
        with open(f"de-en/{file}.de-en.{l}.xml") as f:
            lines = f.read().split("\n")
        cleaned = []
        for line in lines:
            if not '<seg id' in line: continue
            line = re.sub("<seg id=\"\d*\">", "", line)
            line = re.sub("<\/seg>", "", line)
            cleaned.append(line)
        with open(f"tmp.test.{l}", "a") as f:
            f.write("\n".join(cleaned))

for file in ['IWSLT14.TED.dev2010', 'IWSLT14.TEDX.dev2012']:
    for l in ['de', 'en']:
        with open(f"de-en/{file}.de-en.{l}.xml") as f:
            lines = f.read().split("\n")
        cleaned = []
        for line in lines:
            if not '<seg id' in line: continue
            line = re.sub("<seg id=\"\d*\">", "", line)
            line = re.sub("<\/seg>", "", line)
            cleaned.append(line)
        with open(f"tmp.valid.{l}", "a") as f:
            f.write("\n".join(cleaned))

# Tokenize and clean test/validation data
for s in ["test", "valid"]:
    for l in ["de", "en"]:
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l {l} < tmp.{s}.{l} > tmp.tok.{l}")
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < tmp.tok.{l} > tmp.{s}.{l}")


# TODO Apply Joint Dropout

# Learn BPE
num_bpe_tokens: int = 10000
os.system(f"subword-nmt learn-joint-bpe-and-vocab --input tmp.train.en tmp.train.de -s {num_bpe_tokens} -o code.txt --write-vocabulary vocab.en vocab.de")

# Apply BPE
for s in ["train", "test", "valid"]:
    for l in ["de", "en"]:
        os.system(f"subword-nmt apply-bpe -c code.txt --vocabulary vocab.{l} < tmp.{s}.{l} > {s}.{l}")
        os.remove(f"tmp.{s}.{l}")
