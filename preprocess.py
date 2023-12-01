import os
import re
import random
import utils
import subprocess

class Preprocess:

    def __init__(self):
        pass

    def _read_and_clean(self, infile: str, outfile: str):
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

    def preprocess(self,):
        # Configuration Constants
        DATA_DIR = "data/"
        TEMP_DIR = "temp/"
        NUM_THREADS = 1

        # Delete data from old runs in "data" and "temp" directories
        for dir_path in [TEMP_DIR, DATA_DIR]:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                os.system(f"rm -rf {dir_path}")

        # Make temp and data directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('temp', exist_ok=True)

        # Sample source and target
        self._read_and_clean("de-en/train.tags.de-en.en", TEMP_DIR+"tmp.en")
        self._read_and_clean("de-en/train.tags.de-en.de", TEMP_DIR+"tmp.de")

        # Tokenize the data
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l en \
                    < {TEMP_DIR}tmp.en > {TEMP_DIR}tmp.tok.en")
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l de \
                    < {TEMP_DIR}tmp.de > {TEMP_DIR}tmp.tok.de")

        # Clean the data
        os.system(f"perl mosesdecoder/scripts/training/clean-corpus-n.perl {TEMP_DIR}tmp.tok de en \
                    {TEMP_DIR}tmp.clean 1 175")

        # Truecase (lowercase) the data
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.en > \
                    {TEMP_DIR}tmp.train.en")
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.de > \
                    {TEMP_DIR}tmp.train.de")

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
                os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads \
                            {NUM_THREADS} -l {l} < {TEMP_DIR}tmp.{s}.{l} > {TEMP_DIR}tmp.tok.{l}")
                os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl \
                            < {TEMP_DIR}tmp.tok.{l} > {TEMP_DIR}tmp.{s}.{l}")

        # Learn BPE
        num_bpe_tokens: int = 10000
        os.system(f"subword-nmt learn-joint-bpe-and-vocab --input {TEMP_DIR}tmp.train.en \
                    {TEMP_DIR}tmp.train.de -s {num_bpe_tokens} -o {TEMP_DIR}code.txt \
                    --write-vocabulary {TEMP_DIR}vocab.en {TEMP_DIR}vocab.de")
        for s in ["test", "valid"]:
            for l in ["de", "en"]:
                os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.{l} < \
                            {TEMP_DIR}tmp.{s}.{l} > {DATA_DIR}{s}.{l} --glossaries '<S_\d+>' '<T_\d+>'")

    def generate_binarized_data(self, data_dir:str, src_lang:str, trg_lang:str):
        DATA_DIR = "data/"
        TEMP_DIR = "temp/"

        # Apply bpe to train files only
        for file in [f'{DATA_DIR}/train.de',f'{DATA_DIR}/train.en']:
            if os.path.exists(file):
                subprocess.run(["rm", "-rf", file], capture_output=True, text=True) 
        
        s='train'
        for l in ["de", "en"]:
            os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.{l} < \
                        {TEMP_DIR}tmp.{s}.{l} > {DATA_DIR}{s}.{l} --glossaries '<S_\d+>' '<T_\d+>'")
        
        # Create binarized train, test, valid files
        if os.path.exists("data-bin/") and os.path.isdir("data-bin/"):
            subprocess.run(["rm", "-rf", "data-bin"], capture_output=True, text=True)
        subprocess.run(f"""
            fairseq-preprocess --source-lang {src_lang} --target-lang {trg_lang} \
                --trainpref data/train --validpref data/valid --testpref data/test \
                --destdir {data_dir} \
                --workers 20
            """, capture_output=True, text=True, shell=True)


    def preprocess_training_data_only(self,):
        # Configuration Constants
        DATA_DIR = "data/"
        TEMP_DIR = "temp/"
        NUM_THREADS = 1

        # Tokenize the data
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l en \
                    < {TEMP_DIR}tmp.en > {TEMP_DIR}tmp.tok.en")
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l de \
                    < {TEMP_DIR}tmp.de > {TEMP_DIR}tmp.tok.de")

        # Clean the data
        os.system(f"perl mosesdecoder/scripts/training/clean-corpus-n.perl {TEMP_DIR}tmp.tok de en \
                    {TEMP_DIR}tmp.clean 1 175")

        # Truecase (lowercase) the data
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.en > \
                    {TEMP_DIR}tmp.train.en")
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.de > \
                    {TEMP_DIR}tmp.train.de")
