import os
import platform
import random
import re
import shutil
import subprocess
import tarfile
import urllib.request
from indicnlp.tokenize import indic_tokenize
from string import punctuation

class WordAligner:
    def __init__(self):
        self.system_name = platform.system()
        self.eflomal_installed = False
        if os.path.isdir('maceflomal'):
            self.eflomal_installed = True

        if self.system_name == "Darwin":
            subprocess.run(["pip", "install", "eflomal"], capture_output=True, text=True)

        if not self.eflomal_installed:
            if self.system_name == "Darwin":
                self.__setup_macos()
                self.eflomal_installed = True
            else:
                self.__setup_other()
                self.eflomal_installed = True

    def __setup_macos(self):
        # Install Homebrew if not already installed
        if subprocess.run(["which", "brew"]).returncode != 0:
            subprocess.run(["/bin/bash", "-c", "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"], check=True)

        # Install LLVM if not already installed
        if subprocess.run(["brew", "list", "llvm"]).returncode != 0:
            subprocess.run(["brew", "install", "llvm"], check=True)

        subprocess.run(["git", "clone", "https://github.com/huseinzol05/maceflomal"], check=True)
        self.__fix_macos_makefile()
        subprocess.run("cd maceflomal && export CC=/usr/local/bin/gcc-11 && make", shell=True, check=True)
        subprocess.run(["sudo", "make", "install"], cwd="maceflomal", check=True)
        subprocess.run(["python3", "setup.py", "install"], cwd="maceflomal", check=True)

    def __fix_macos_makefile(self):
        # Define the directory of the Makefile
        makefile_dir = os.path.join(os.getcwd(), "maceflomal")
        makefile_path = os.path.join(makefile_dir, "Makefile")

        # Read the Makefile
        with open(makefile_path, "r") as f:
            content = f.read()

        # Replace spaces with tabs
        content = content.replace("    ", "\t")

        # Make other replacements
        content = content.replace("CFLAGS=-Ofast -march=native -Wall --std=gnu99 -Wno-unused-function -g -fopenmp",
                                  "CC=/opt/homebrew/Cellar/llvm/17.0.3/bin/clang\nCFLAGS=-Ofast -Wall --std=gnu99 -Wno-unused-function -g -fopenmp")
        content = content.replace("LDFLAGS=-lm -lgomp -fopenmp",
                                  "LDFLAGS=-L/opt/homebrew/Cellar/llvm/17.0.3/lib -Wl,-rpath,/opt/homebrew/Cellar/llvm/17.0.3/lib -lm -lomp -fopenmp")

        # Write the modified content back to the Makefile
        with open(makefile_path, "w") as f:
            f.write(content)

    def __setup_other(self):
        subprocess.run(["pip", "install", "eflomal==1.0.0b0"], capture_output=True, text=True)
        global eflomal_module
        eflomal_module = __import__('eflomal')

    def word_alignments(self, source_file, target_file, output_file, model='3'):
        if self.system_name == "Darwin":
            subprocess.run([
                "python", os.path.join("maceflomal/", "align.py"),
                "-s", source_file,
                "-t", target_file,
                "-f", output_file,
                "--model", model,
                "--source-prefix", "0",
                "--source-suffix", "0",
                "--target-prefix", "0",
                "--target-suffix", "0",
                "--overwrite"
            ], check=True)
        else:
            aligner = eflomal_module.Aligner(model = int(model)) 

            with open(source_file, 'r', encoding='utf-8') as src_data, \
                 open(target_file, 'r', encoding='utf-8') as trg_data:
                aligner.align(src_input = src_data,
                              trg_input = trg_data,
                              links_filename_fwd = output_file)

class JointDropout:
    def __init__(self, debug=False):
        self.debug = debug

    # Phrase Extraction Algorithm
    # Original version from NLTK: http://goo.gl/ZLKexJ
    # Authors: Liling Tan
    def _phrase_extraction(self, srctext, trgtext, alignment):
        """
        Phrase extraction algorithm extracts all consistent phrase pairs from 
        a word-aligned sentence pair.

        The idea is to loop over all possible source language (e) phrases and find 
        the minimal foregin phrase (f) that matches each of them. Matching is done 
        by identifying all alignment points for the source phrase and finding the 
        shortest foreign phrase that includes all hte foreign counterparts for the 
        source words.

        In short, a phrase alignment has to 
        (a) contain all alignment points for all covered words
        (b) contain at least one alignment point

        A phrase pair (e, f ) is consistent with an alignment A if and only if:
        
        (i) No English words in the phrase pair are aligned to words outside it.
        
               ∀e i ∈ e, (e i , f j ) ∈ A ⇒ f j ∈ f
        
        (ii) No Foreign words in the phrase pair are aligned to words outside it. 
                
                ∀f j ∈ f , (e i , f j ) ∈ A ⇒ e i ∈ e
        
        (iii) The phrase pair contains at least one alignment point. 
                
                ∃e i ∈ e  ̄ , f j ∈ f  ̄ s.t. (e i , f j ) ∈ A
                
        [in]:
        *srctext* is the tokenized source sentence string.
        *trgtext* is the tokenized target sentence string.
        *alignment* is the word alignment outputs in pharaoh format
        
        [out]:
        *bp* is the phrases extracted from the algorithm, it's made up of a tuple 
        that stores:
            ( (src_from, src_to), (trg_from, trg_to), src_phrase, target_phrase )
        
        (i)   the position of the source phrase
        (ii)  the position of the target phrase
        (iii) the source phrase
        (iv)  the target phrase

        >>> srctext = "michael assumes that he will stay in the house"
        >>> trgtext = "michael geht davon aus , dass er im haus bleibt"
        >>> alignment = [(0,0), (1,1), (1,2), (1,3), (2,5), (3,6), (4,9), 
        ... (5,9), (6,7), (7,7), (8,8)]
        >>> phrases = phrase_extraction(srctext, trgtext, alignment)
        >>> for i in sorted(phrases):
        ...    print i
        ...
        ((0, 1), (0, 1), 'michael', 'michael')
        ((0, 2), (0, 4), 'michael assumes', 'michael geht davon aus')
        ((0, 2), (0, 4), 'michael assumes', 'michael geht davon aus ,')
        ((0, 3), (0, 6), 'michael assumes that', 'michael geht davon aus , dass')
        ((0, 4), (0, 7), 'michael assumes that he', 'michael geht davon aus , dass er')
        ((0, 9), (0, 10), 'michael assumes that he will stay in the house', 'michael geht davon aus , dass er im haus bleibt')
        ((1, 2), (1, 4), 'assumes', 'geht davon aus')
        ((1, 2), (1, 4), 'assumes', 'geht davon aus ,')
        ((1, 3), (1, 6), 'assumes that', 'geht davon aus , dass')
        ((1, 4), (1, 7), 'assumes that he', 'geht davon aus , dass er')
        ((1, 9), (1, 10), 'assumes that he will stay in the house', 'geht davon aus , dass er im haus bleibt')
        ((2, 3), (5, 6), 'that', ', dass')
        ((2, 3), (5, 6), 'that', 'dass')
        ((2, 4), (5, 7), 'that he', ', dass er')
        ((2, 4), (5, 7), 'that he', 'dass er')
        ((2, 9), (5, 10), 'that he will stay in the house', ', dass er im haus bleibt')
        ((2, 9), (5, 10), 'that he will stay in the house', 'dass er im haus bleibt')
        ((3, 4), (6, 7), 'he', 'er')
        ((3, 9), (6, 10), 'he will stay in the house', 'er im haus bleibt')
        ((4, 6), (9, 10), 'will stay', 'bleibt')
        ((4, 9), (7, 10), 'will stay in the house', 'im haus bleibt')
        ((6, 8), (7, 8), 'in the', 'im')
        ((6, 9), (7, 9), 'in the house', 'im haus')
        ((8, 9), (8, 9), 'house', 'haus')
        """
        def extract(f_start, f_end, e_start, e_end):
            if f_end < 0:  # 0-based indexing.
                return {}
            # Check if alignement points are consistent.
            for e,f in alignment:
                if ((f_start <= f <= f_end) and
                   (e < e_start or e > e_end)):
                    return {}

            # Add phrase pairs (incl. additional unaligned f)
            # Remark:  how to interpret "additional unaligned f"?
            phrases = set()
            fs = f_start
            # repeat-
            while True:
                fe = f_end
                # repeat-
                while True:
                    # add phrase pair ([e_start, e_end], [fs, fe]) to set E
                    # Need to +1 in range  to include the end-point.
                    src_phrase = " ".join(srctext[i] for i in range(e_start,e_end+1))
                    trg_phrase = " ".join(trgtext[i] for i in range(fs,fe+1))
                    # Include more data for later ordering.
                    phrases.add(((e_start, e_end+1), (f_start, f_end+1), src_phrase, trg_phrase))
                    fe += 1 # fe++
                    # -until fe aligned or out-of-bounds
                    if fe in f_aligned or fe == trglen:
                        break
                fs -=1  # fe--
                # -until fs aligned or out-of- bounds
                if fs in f_aligned or fs < 0:
                    break
            return phrases

        # Calculate no. of tokens in source and target texts.
        srctext = srctext.split()   # e
        trgtext = trgtext.split()   # f
        srclen = len(srctext)       # len(e)
        trglen = len(trgtext)       # len(f)
        # Keeps an index of which source/target words are aligned.
        e_aligned = [i for i,_ in alignment]
        f_aligned = [j for _,j in alignment]

        bp = set() # set of phrase pairs BP
        # for e start = 1 ... length(e) do
        # Index e_start from 0 to len(e) - 1
        for e_start in range(srclen):
            # for e end = e start ... length(e) do
            # Index e_end from e_start to len(e) - 1
            for e_end in range(e_start, srclen):
                # // find the minimally matching foreign phrase
                # (f start , f end ) = ( length(f), 0 )
                # f_start ∈ [0, len(f) - 1]; f_end ∈ [0, len(f) - 1]
                f_start, f_end = trglen-1 , -1  #  0-based indexing
                # for all (e,f) ∈ A do
                for e,f in alignment:
                    # if e start ≤ e ≤ e end then
                    if e_start <= e <= e_end:
                        f_start = min(f, f_start)
                        f_end = max(f, f_end)
                # add extract (f start , f end , e start , e end ) to set BP
                phrases = extract(f_start, f_end, e_start, e_end)
                if phrases:
                    bp.update(phrases)
        return bp

    def _trim_phrase_translation_table(self, phrase_table):
        def phrase_length(phrase):
            return phrase[0][1] - phrase[0][0], phrase[1][1] - phrase[1][0]

        sorted_phrases = sorted(phrase_table, key=phrase_length)

        filtered_phrases = []
        for phrase in sorted_phrases:
            src_length, tgt_length = phrase_length(phrase)

            if src_length <= 3 and tgt_length <= 3:
                overlap = False
                for kept_phrase in filtered_phrases:
                    if (phrase[0][0] < kept_phrase[0][1] and phrase[0][1] > kept_phrase[0][0]) or \
                       (phrase[1][0] < kept_phrase[1][1] and phrase[1][1] > kept_phrase[1][0]):
                        overlap = True
                        break
                if not overlap:
                    filtered_phrases.append(phrase)
        filtered_phrase_table = {entry for entry in filtered_phrases if not \
                any(char in punctuation for char in entry[2]) and not any(char in punctuation for char in entry[3])}
        return set(filtered_phrase_table)

    def _joint_dropout(self, phrase_table, source_sentence, target_sentence, target_jdr=0.3, max_drops=10):
        source_tokens = source_sentence.split()
        target_tokens = target_sentence.split()
        total_tokens = len(source_tokens) + len(target_tokens)
        dropped_tokens = 0
        drops = 0
        
        # Convert set to list and shuffle for randomness
        phrases = list(phrase_table)
        random.shuffle(phrases)

        dropped_phrases = []
        var_index = 0
        # Select candidate phrase to be dropped
        if len(phrases) == 0:
            return ' '.join(source_tokens), ' '.join(target_tokens), total_tokens, dropped_tokens
        (src_start, src_end), (tgt_start, tgt_end), src_phrase, tgt_phrase = phrases.pop()
        while phrases:
            # Check if phrase violated adjacency clause (cannot have adjacent variables)
            if len(dropped_phrases) != 0:
                not_valid_phrase = True
                while not_valid_phrase:
                    if len(phrases) == 0:
                        return ' '.join(source_tokens), ' '.join(target_tokens), total_tokens, dropped_tokens
                    (src_start, src_end), (tgt_start, tgt_end), src_phrase, tgt_phrase = phrases.pop() 
                    for i, ((dropped_src_start, dropped_src_end), (dropped_tgt_start, dropped_tgt_end),
                            dropped_src_phrase, dropped_tgt_phrase) in enumerate(dropped_phrases):
                        # If candidate phrase and is adjacent with dropped phrase
                        if src_end == dropped_src_start or tgt_end == dropped_tgt_start or \
                                dropped_src_end == src_start or dropped_tgt_end == tgt_start:
                            break

                        # If there are no adjaced dropped phrases to current phrase, proceed
                        if i == (len(dropped_phrases) - 1):
                            not_valid_phrase = False

            # Update counters and sentences with distinct variable names enclosed in <>
            dropped_tokens += (src_end - src_start) + (tgt_end - tgt_start)
            source_tokens[src_start:src_end] = [f"<S_{var_index}>"]
            target_tokens[tgt_start:tgt_end] = [f"<T_{var_index}>"]

            dropped_phrases.append(((src_start, src_end), (tgt_start, tgt_end), src_phrase, tgt_phrase))
            drops += 1
            var_index += 1

            # Check for stopping conditions
            jdr = dropped_tokens / total_tokens
            if jdr >= target_jdr or drops >= max_drops:
                break
        return ' '.join(source_tokens), ' '.join(target_tokens), total_tokens, dropped_tokens

    def joint_dropout(self, src_file, trg_file, word_alignment_file, output_dir='./',
                      src_suffix='out', trg_suffix='out', target_jdr=0.3):
        with open(src_file,'r') as src_f, \
             open(trg_file,'r') as trg_f, \
             open(word_alignment_file,'r') as algn_f, \
             open(output_dir+'jdr.src.'+src_suffix, 'w') as trg_out_f, \
             open(output_dir+'jdr.trg.'+trg_suffix, 'w') as src_out_f:
                tokens_dropped = 0
                tokens_total = 0
                i=0
                for src_line, trg_line, algn_line in zip(src_f, trg_f, algn_f):
                    algn_list = algn_line.split()
                    algn_list = [(int(s.split('-')[0]), int(s.split('-')[1])) for s in algn_list]
                    ptt = self._phrase_extraction(src_line, trg_line, algn_list)
                    ptt = self._trim_phrase_translation_table(ptt)
                    source_sentence, target_sentence, total_tokens, dropped_tokens = \
                            self._joint_dropout(ptt, src_line, trg_line, target_jdr=0.3, max_drops=10)
                    
                    tokens_dropped += dropped_tokens
                    tokens_total += total_tokens
                    trg_out_f.write(source_sentence+'\n')
                    src_out_f.write(target_sentence+'\n')
                    if self.debug: print(source_sentence)
                    if self.debug: print(target_sentence+'\n')
                    i += 1

                if self.debug: print(f"JDR: {tokens_dropped / tokens_total}")

def _read_and_clean(infile: str, outfile: str):
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

def download_de_en_data(tmp_dir: str):

    if not os.path.exists("de-en"):
        urllib.request.urlretrieve("http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz", "de-en.tgz")
        file = tarfile.open("de-en.tgz")
        file.extractall()
        os.remove("de-en.tgz")

    # Sample source and target
    _read_and_clean(f"de-en/train.tags.de-en.en", tmp_dir+f"tmp.en")
    _read_and_clean(f"de-en/train.tags.de-en.de", tmp_dir+f"tmp.de")

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
            with open(f"{tmp_dir}tmp.test.{l}", "a") as f:
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
            with open(f"{tmp_dir}tmp.valid.{l}", "a") as f:
                f.write("\n".join(cleaned))

def develop_ne_en_data(tmp_dir: str):
    # Download data
    if os.path.exists("ne-en"):
        shutil.rmtree("ne-en")

    urllib.request.urlretrieve("https://raw.githubusercontent.com/facebookresearch/flores/main/previous_releases/floresv1/data/wikipedia_en_ne_si_test_sets.tgz", "test_sets.tgz")
    file = tarfile.open("test_sets.tgz")
    file.extractall()
    os.remove("test_sets.tgz")

    os.mkdir("ne-en")

    for src, tgt in [('devtest', 'valid'), ('test', 'test')]:
        for l in ['ne', 'en']:
            outfile = os.path.join("ne-en", f"{tgt}.{l}")
            os.system(f"cp wikipedia_en_ne_si_test_sets/wikipedia.{src}.ne-en.{l} {outfile}")
    
    shutil.rmtree("wikipedia_en_ne_si_test_sets")

    # Training data
    os.system("bash ./download-data.sh")

    train_sets = [
        'clean-data/all-clean-ne/bible.en-ne',
        'clean-data/all-clean-ne/bible_dup.en-ne',
        'clean-data/all-clean-ne/GlobalVoices.en-ne',
        'clean-data/all-clean-ne/GNOMEKDEUbuntu.en-ne',
        'clean-data/all-clean-ne/nepali-penn-treebank',
    ]
    for l in ['ne', 'en']:
        files = [f"{s}.{l}" for s in train_sets]
        outfile = os.path.join("ne-en", f"train.{l}")
        os.system(f"cat {' '.join(files)} > {outfile}")
    
    shutil.rmtree("clean-data")
    
    # Preprocess data
    for l in ["en", "ne"]:
        for s in ["train", "test", "valid"]:
            src = os.path.join("ne-en", f"{s}.{l}")
            tgt = f"{tmp_dir}tmp.{s}.{l}"
            shutil.copyfile(src, tgt)    

def download_data(tmp_dir: str, src_lang: str, tgt_lang: str):
    if src_lang == "de" and tgt_lang == "en":
        download_de_en_data(tmp_dir)
    if src_lang == "ne" and tgt_lang == "en":
        develop_ne_en_data(tmp_dir)

def preprocess_moses(tmp_dir: str, src_lang: str, tgt_lang: str):
    num_threads = 1
    # Tokenize the data
    os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {num_threads} -l {tgt_lang} \
                < {tmp_dir}tmp.{tgt_lang} > {tmp_dir}tmp.tok.{tgt_lang}")
    os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {num_threads} -l {src_lang} \
                < {tmp_dir}tmp.{src_lang} > {tmp_dir}tmp.tok.{src_lang}")

    # Clean the data
    os.system(f"perl mosesdecoder/scripts/training/clean-corpus-n.perl {tmp_dir}tmp.tok {src_lang} {tgt_lang} {tmp_dir}tmp.clean 1 175")

    # Truecase (lowercase) the data
    os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {tmp_dir}tmp.clean.{src_lang} > {tmp_dir}tmp.train.{src_lang}")
    os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {tmp_dir}tmp.clean.{tgt_lang} > {tmp_dir}tmp.train.{tgt_lang}")

    # Tokenize and clean test/validation data
    for s in ["test", "valid"]:
        for l in [src_lang, tgt_lang]:
            os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {num_threads} -l {l} \
                        < {tmp_dir}tmp.{s}.{l} > {tmp_dir}tmp.tok.{l}")
            os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {tmp_dir}tmp.tok.{l} > \
                        {tmp_dir}tmp.{s}.{l}")

    # Sample the data
    no_samples = 10000
    with open(f"{tmp_dir}tmp.train.{tgt_lang}") as f:
        train_tgt = f.read().split("\n")
    with open(f"{tmp_dir}tmp.train.{src_lang}") as f:
        train_src = f.read().split("\n")
    random.seed(1)
    samples = random.sample(range(len(train_tgt)), no_samples)
    train_tgt = [train_tgt[i] for i in samples]
    train_src = [train_src[i] for i in samples]
    with open(f"{tmp_dir}tmp.train.{src_lang}", "w") as f:
        f.write("\n".join(train_src))
    with open(f"{tmp_dir}tmp.train.{tgt_lang}", "w") as f:
        f.write("\n".join(train_tgt))

def preprocess_indic(tmp_dir: str, src_lang: str, tgt_lang: str):
    # Process source language using Indic NLP Library
    with open(f"{tmp_dir}tmp.test.{src_lang}") as f:
        tgt = f.readlines()
    # Tokenize
    tgt = [indic_tokenize.trivial_tokenize(s, src_lang) for s in tgt]
    # Concatenate - DOES THIS EVEN HAVE A PURPOSE?
    tgt = [" ".join(s) for s in tgt]
    with open(f"{tmp_dir}tmp.test.{src_lang}", "w") as f:
        tgt = f.write("".join(tgt))

    # Process target language using moses
    num_threads = 1
    for s in ["train", "test", "valid"]:
        os.system(f"perl mosesdecoder/scripts/tokenizer/tokenizer.perl -threads {num_threads} -l en \
                    < {tmp_dir}tmp.{s}.{tgt_lang} > {tmp_dir}tmp.tok.{tgt_lang}")
        os.system(f"perl mosesdecoder/scripts/tokenizer/lowercase.perl < {tmp_dir}tmp.tok.{tgt_lang} > \
                    {tmp_dir}tmp.{s}.{tgt_lang}")

    # Sample the data
    no_samples = 20000
    with open(f"{tmp_dir}tmp.train.{tgt_lang}") as f:
        train_tgt = f.read().split("\n")
    with open(f"{tmp_dir}tmp.train.{src_lang}") as f:
        train_src = f.read().split("\n")
    random.seed(1)
    samples = random.sample(range(len(train_tgt)), no_samples)
    train_tgt = [train_tgt[i] for i in samples]
    train_src = [train_src[i] for i in samples]
    with open(f"{tmp_dir}tmp.train.{src_lang}", "w") as f:
        f.write("\n".join(train_src))
    with open(f"{tmp_dir}tmp.train.{tgt_lang}", "w") as f:
        f.write("\n".join(train_tgt))

def preprocess_data(tmp_dir: str, src_lang: str, tgt_lang: str):
    if (src_lang == "de" and tgt_lang == "en"):
        preprocess_moses(tmp_dir, src_lang, tgt_lang)
    if (src_lang == "ne" and tgt_lang == "en"):
        preprocess_indic(tmp_dir, src_lang, tgt_lang)