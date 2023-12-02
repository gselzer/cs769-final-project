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

    def preprocess(self, src_lang:str, trg_lang:str):
        # Configuration Constants
        DATA_DIR = "data/"
        TEMP_DIR = "temp/"

        # Delete data from old runs in "data" and "temp" directories
        for dir_path in [TEMP_DIR, DATA_DIR]:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                os.system(f"rm -rf {dir_path}")

        # Make temp and data directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('temp', exist_ok=True)

        utils.download_data(TEMP_DIR, src_lang, trg_lang)
        utils.preprocess_data(TEMP_DIR, src_lang, trg_lang)

        # Learn BPE
        num_bpe_tokens: int = 10000
        os.system(f"subword-nmt learn-joint-bpe-and-vocab --input {TEMP_DIR}tmp.train.{trg_lang} \
                    {TEMP_DIR}tmp.train.{src_lang} -s {num_bpe_tokens} -o {TEMP_DIR}code.txt \
                    --write-vocabulary {TEMP_DIR}vocab.{trg_lang} {TEMP_DIR}vocab.{src_lang}")
        for s in ["test", "valid"]:
            for l in [src_lang, trg_lang]:
                os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.{l} < \
                            {TEMP_DIR}tmp.{s}.{l} > {DATA_DIR}{s}.{l} --glossaries '<S_\d+>' '<T_\d+>'")

    def generate_binarized_data(self, data_dir:str, src_lang:str, trg_lang:str):
        DATA_DIR = "data/"
        TEMP_DIR = "temp/"

        # Apply bpe to train files only
        for file in [f'{DATA_DIR}/train.{src_lang}',f'{DATA_DIR}/train.{trg_lang}']:
            if os.path.exists(file):
                subprocess.run(["rm", "-rf", file], capture_output=True, text=True) 
        
        s='train'
        for l in [src_lang, trg_lang]:
            os.system(f"subword-nmt apply-bpe -c {TEMP_DIR}code.txt --vocabulary {TEMP_DIR}vocab.{l} < \
                        {TEMP_DIR}tmp.{s}.{l} > {DATA_DIR}{s}.{l} --glossaries '<S_\d+>' '<T_\d+>'")
        
        # Create binarized train, test, valid files
        subprocess.run(f"""
            fairseq-preprocess --source-lang {src_lang} --target-lang {trg_lang} \
                --trainpref data/train --validpref data/valid --testpref data/test \
                --destdir {data_dir} \
                --workers 20
            """, capture_output=True, text=True, shell=True)


    def preprocess_training_data_only(self, src_lang:str, trg_lang:str):
        # Configuration Constants
        DATA_DIR = "data/"
        TEMP_DIR = "temp/"
        NUM_THREADS = 1

        # Tokenize the data
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l \
                    {src_lang} < {TEMP_DIR}tmp.{src_lang} > {TEMP_DIR}tmp.tok.{src_lang}")
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {NUM_THREADS} -l \
                    {trg_lang} < {TEMP_DIR}tmp.{trg_lang} > {TEMP_DIR}tmp.tok.{trg_lang}")

        # Clean the data
        os.system(f"perl mosesdecoder/scripts/training/clean-corpus-n.perl {TEMP_DIR}tmp.tok \
                    {src_lang} {trg_lang} {TEMP_DIR}tmp.clean 1 175")

        # Truecase (lowercase) the data
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl <{TEMP_DIR}tmp.clean.{src_lang} > \
                    {TEMP_DIR}tmp.train.{src_lang}")
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {TEMP_DIR}tmp.clean.{trg_lang} > \
                    {TEMP_DIR}tmp.train.{trg_lang}")
