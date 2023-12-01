import subprocess
import preprocess
import os
import time

TRANSLATIONS_DIR = "translations/"
TEMP_DIR = "temp/"
DEBUG = True
VERBOSE = True # set to False if you want to hide the outputs of fairseq-train and fairseq-generate

class DataDiversification:

    def __init__(self, k=3, N=1, n_epoch=80):
        """
        Parameters:
            k: (int) diversification factor (default = 3)
            N: (int) number of rounds (default = 1)
            n_epoch: (int) number of epochs to train models for
        """
        self.k = k
        self.N = N
        self.n_epoch = n_epoch

    def _train(self, data_dir:str, arch:str, src_lang:str, trg_lang:str, suffix:str):
        if os.path.exists("models/") and os.path.isdir("models/"):
            subprocess.run("rm -rf models", capture_output=True, text=True, shell=True)
        subprocess.run("mkdir models", capture_output=True, text=True, shell=True)

        subprocess.run(f"""
            CUDA_VISIBLE_DEVICES=0 fairseq-train \
                {data_dir} \
                --arch {arch} --share-decoder-input-output-embed \
                --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                --source-lang {src_lang} --target-lang {trg_lang} \
                --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
                --dropout 0.3 --weight-decay 0.0001 \
                --max-epoch {self.n_epoch} \
                --criterion label_smoothed_cross_entropy --label-smoothing 0.5 \
                --max-tokens 4096 \
                --scoring sacrebleu \
                --eval-bleu \
                --eval-bleu-args '{{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}}' \
                --eval-bleu-detok moses \
                --eval-bleu-remove-bpe \
                --eval-bleu-print-samples \
                --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
                --checkpoint-suffix {suffix} --no-last-checkpoints \
                --encoder-layers 5 --decoder-layers 5 \
                --encoder-attention-heads 2 --decoder-attention-heads 2 \
                --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
                --encoder-layerdrop 0.0 --decoder-layerdrop 0.2 \
                --activation-dropout 0.3 \
                --save-interval {self.n_epoch} \
                --validate-interval {self.n_epoch} \
                --save-dir models
            """, text=True, shell=True, capture_output= (not VERBOSE))
            # --ddp-backend=legacy_ddp \

        if DEBUG:
            file_path = f'models/checkpoint_best{suffix}.pt'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"'_train' failed to generate the model '{file_path}'.")

    def _generate(self, data_dir: str, src_lang: str, trg_lang: str, suffix: str):
        for file in ['output.txt', 'outputfile.txt']:
            if os.path.exists(file):
                subprocess.run(f"rm -rf {file}", capture_output=True, text=True, shell=True)
    
        subprocess.run(f"""
            fairseq-generate {data_dir} \
                --path models/checkpoint_best{suffix}.pt \
                --beam 5 \
                --max-tokens 4096 \
                --source-lang {src_lang} --target-lang {trg_lang} \
                --tokenizer moses \
                --sacrebleu \
                --gen-subset train \
                --post-process subword_nmt \
                >> output.txt
        """, text=True, shell=True, capture_output = (not VERBOSE))
        #  --no-progress-bar \
        # --batch-size 128 \

        if DEBUG:
            file_path = 'output.txt'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"'_generate' failed to generate intermediate file '{file_path}'.")

            line_count = sum(1 for line in open('output.txt'))
            if line_count == 0:
                raise ValueError("'_generate' created '{file_path}' with zero lines.")

        subprocess.run(f"grep -E '^S-[0-9]+|^T-[0-9]+' output.txt > outputfile.txt",
                       capture_output=True, text=True, shell=True)

        if DEBUG:
            file_path = 'outputfile.txt'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"'_generate' failed to generate the file '{file_path}'.")

            line_count = sum(1 for line in open('outputfile.txt'))
            if line_count == 0:
                raise ValueError(f"'_generate' created '{file_path}' with zero lines.") 

    def _append_bitext(self, input_file:str, source_file:str, target_file:str):
        with open(input_file, 'r') as infile, \
             open(source_file, 'a') as srcfile, \
             open(target_file, 'a') as tgtfile:
            
            for line in infile:
                if line.startswith('S-'):
                    content_start = line.find('\t') + 1  
                    srcfile.write(line[content_start:])  
                elif line.startswith('T-'):
                    content_start = line.find('\t') + 1  
                    tgtfile.write(line[content_start:]) 

    def diversify(self, arch_fwd:str, arch_bkwd:str, src_lang:str, trg_lang:str):
        """
        Performs the data diversification.
        """

        # Preprocess initial dataset
        preprocessObj = preprocess.Preprocess()
        preprocessObj.preprocess()
       
        if os.path.exists(f"{TRANSLATIONS_DIR}"):
            subprocess.run(f"rm -rf {TRANSLATIONS_DIR}", capture_output=True, text=True, shell=True)
        subprocess.run(f"mkdir {TRANSLATIONS_DIR}", capture_output=True, text=True, shell=True)

        # Copy original dataset to translations directory
        subprocess.run(f"cp temp/tmp.train.{src_lang} {TRANSLATIONS_DIR}/train.{src_lang}", 
                        capture_output=True, text=True, shell=True)
        subprocess.run(f"cp temp/tmp.train.{trg_lang} {TRANSLATIONS_DIR}/train.{trg_lang}", 
                        capture_output=True, text=True, shell=True)

        for i in range(1,self.N+1):
            # Binarize data for both forward and backward models
            preprocessObj.generate_binarized_data(
                data_dir='data/binarized-fwd', 
                src_lang=src_lang, trg_lang=trg_lang)
            preprocessObj.generate_binarized_data(
                data_dir='data/binarized-bkwd', 
                src_lang=trg_lang, trg_lang=src_lang)

            for j in range(1,self.k+1):
                
                # Train M_f on D_r-1
                start_time = time.time()
                print(f"\nBeginning training of forward translation model {i}.{j}")
                self._train(
                    data_dir='data/binarized-fwd',
                    arch=arch_fwd, 
                    src_lang=src_lang, 
                    trg_lang=trg_lang, 
                    suffix='_fwd')
                print(f"Finished training of forward translation model {i}.{j} in \
                        {(time.time() - start_time)/60:.2f} minutes")
                
                # Generate M_f(S)
                start_time = time.time()
                print(f"\nBeginning generating M_f{i}.{j}(S)")
                self._generate(
                    data_dir='data/binarized-fwd', 
                    src_lang=src_lang, 
                    trg_lang=trg_lang, 
                    suffix='_fwd')
                print(f"Finished generating M_f{i}.{j}(S) in {(time.time() - start_time)/60:.2f} minutes")
                
                # Append (S, M_f(S)) to D_r
                self._append_bitext("outputfile.txt", f"{TRANSLATIONS_DIR}/train.{src_lang}", 
                                    f"{TRANSLATIONS_DIR}/train.{trg_lang}")
                
                # Train M_b on D_r-1'
                start_time = time.time()
                print(f"\nBeginning training of backward translation model {i}.{j}")
                self._train(
                    data_dir='data/binarized-bkwd', 
                    arch=arch_bkwd, 
                    src_lang=trg_lang, 
                    trg_lang=src_lang,
                    suffix='_bkwd')
                print(f"Finished training of backward translation model {i}.{j} in \
                        {(time.time() - start_time)/60:.2f} minutes")

                # Generate M_b(T) 
                start_time = time.time()
                print(f"\nBeginning generating M_b{i}.{j}(T)")
                self._generate(
                    data_dir='data/binarized-bkwd', 
                    src_lang=trg_lang, 
                    trg_lang=src_lang,
                    suffix='_bkwd')
                print(f"Finished generating M_b{i}.{j}(T) in {(time.time() - start_time)/60:.2f} minutes")

                # Append (M_b(T), T) to D_r
                self._append_bitext("outputfile.txt", f"{TRANSLATIONS_DIR}/train.{trg_lang}", 
                                    f"{TRANSLATIONS_DIR}/train.{src_lang}")

            # copy D_r to D_r-1
            for file in ['tmp','tmp.tok','tmp.clean','tmp.train']:
                for l in [src_lang,trg_lang]:
                    subprocess.run(f"rm -rf {TEMP_DIR}{file}.{l}", capture_output=True, text=True,
                                    shell=True)
                    
            subprocess.run(f"cp {TRANSLATIONS_DIR}/train.{src_lang} temp/tmp.{src_lang}", 
                        capture_output=True, text=True, shell=True)
            subprocess.run(f"cp {TRANSLATIONS_DIR}/train.{trg_lang} temp/tmp.{trg_lang}", 
                        capture_output=True, text=True, shell=True)

            # Preprocess training set with its new sythetic data
            preprocessObj.preprocess_training_data_only()

