import os
import re
import random
import utils

from mBART import mbart

# Configuration Constants
DATA_DIR = "data/"
TEMP_DIR = "temp/"
DELETE_TEMP_DATA = True
NUM_THREADS = 1

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

def noising(sentence, p=0.35):
    words = sentence.split()
    
    # Randomly replace some words with [MASK]
    for i in range(len(words)):
        if random.random() < p:
            words[i] = "[MASK]"
    
    # Randomly shuffle the order of words
    random.shuffle(words)
    
    return " ".join(words)

def apply_noising_to_file(infile, outfile, p=0.35):
    with open(infile, "r") as f:
        sentences = f.read().split("\n")
    
    noised_sentences = [noising(sentence, p) for sentence in sentences]
    
    with open(outfile, "w") as f:
        f.write("\n".join(noised_sentences))


# Delete data from old runs in "data" and "temp" directories
for dir_path in [TEMP_DIR, DATA_DIR]:
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        os.system(f"rm -rf {dir_path}")

# Make temp and data directories
os.makedirs('data', exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Sample source and target
read_and_clean("de-en/train.tags.de-en.en", TEMP_DIR+"tmp.en")
read_and_clean("de-en/train.tags.de-en.de", TEMP_DIR+"tmp.de")

# Tokenize the data
os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l en \
            < {TEMP_DIR}tmp.en > {TEMP_DIR}tmp.tok.en")
os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l de \
            < {TEMP_DIR}tmp.de > {TEMP_DIR}tmp.tok.de")

# Clean the data
os.system(f"perl mosesdecoder/scripts/training/clean-corpus-n.perl {TEMP_DIR}tmp.tok de en {TEMP_DIR}tmp.clean 1 175")

mbart(
    [f"{TEMP_DIR}tmp.clean.en", f"{TEMP_DIR}tmp.clean.de"], 
    [f"{TEMP_DIR}tmp.train.en", f"{TEMP_DIR}tmp.train.de"])
# apply_noising_to_file(f"{TEMP_DIR}tmp.clean.en", f"{TEMP_DIR}tmp.train.en")
# apply_noising_to_file(f"{TEMP_DIR}tmp.clean.de", f"{TEMP_DIR}tmp.train.de")


# Truecase (lowercase) the data
# os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.en > {TEMP_DIR}tmp.train.en")
# os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.de > {TEMP_DIR}tmp.train.de")

# Sample the data
no_samples = 10000
with open(TEMP_DIR+"tmp.train.en") as f:
    train_en = f.read().split("\n")
with open(TEMP_DIR+"tmp.train.de") as f:
    train_de = f.read().split("\n")
samples = random.sample(range(len(train_en)), no_samples)
train_en = [train_en[i] for i in samples]
train_de = [train_de[i] for i in samples]
with open(TEMP_DIR+"tmp.train.en", "w") as f:
    f.write("\n".join(train_en))
with open(TEMP_DIR+"tmp.train.de", "w") as f:
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
        with open(f"{TEMP_DIR}tmp.test.{l}", "a") as f:
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
        with open(f"{TEMP_DIR}tmp.valid.{l}", "a") as f:
            f.write("\n".join(cleaned))

# Tokenize and clean test/validation data
for s in ["test", "valid"]:
    for l in ["de", "en"]:
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l {l} \
                    < {TEMP_DIR}tmp.{s}.{l} > {TEMP_DIR}tmp.tok.{l}")
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.tok.{l} > \
                    {TEMP_DIR}tmp.{s}.{l}")

# Learn BPE
num_bpe_tokens: int = 10000
os.system(f"subword-nmt learn-joint-bpe-and-vocab --input {TEMP_DIR}tmp.train.en {TEMP_DIR}tmp.train.de \
            -s {num_bpe_tokens} -o {TEMP_DIR}code.txt --write-vocabulary {TEMP_DIR}vocab.en {TEMP_DIR}vocab.de")

# Joint Dropout
wa = utils.WordAligner()
wa.word_alignments(source_file=f"{TEMP_DIR}tmp.train.de",
                    target_file=f"{TEMP_DIR}tmp.train.en",
                    output_file=f'{TEMP_DIR}eflomal.de.en',
                    model = '3')
JDR = utils.JointDropout(debug=False)
JDR.joint_dropout(f"{TEMP_DIR}tmp.train.de", f"{TEMP_DIR}tmp.train.en", f'{TEMP_DIR}eflomal.de.en',
                  output_dir=f'{TEMP_DIR}', src_suffix='de',trg_suffix='en')

# Concatenate JDR output with the tmp.train.en/tmp.train.de files.
with open(f'{TEMP_DIR}jdr.src.de', 'r') as source_file:
    data_to_append = source_file.read()

with open(f'{TEMP_DIR}tmp.train.de', 'a') as target_file:
    target_file.write(data_to_append)

with open(f'{TEMP_DIR}jdr.trg.en', 'r') as source_file:
    data_to_append = source_file.read()

with open(f'{TEMP_DIR}tmp.train.en', 'a') as target_file:
    target_file.write(data_to_append)

# Apply BPE
for s in ["train", "test", "valid"]:
    for l in ["de", "en"]:
        os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.{l} < \
                    {TEMP_DIR}tmp.{s}.{l} > {DATA_DIR}{s}.{l} --glossaries '<S_\d+>' '<T_\d+>'")

# Delete Temp Files
if DELETE_TEMP_DATA:
    os.system("rm -rf temp")
