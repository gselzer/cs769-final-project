from numpy.random import poisson
import random
import os
from typing import List

def mbart(
        src_dir: str, 
        output_dir: str, 
        src_lang: str,
        tgt_lang: str,
        lambda_value: float = 3.5):
 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Noise training data
    for lang in [src_lang, tgt_lang] :
        file = os.path.join(src_dir, f"tmp.train.{lang}")
        with open(file, "r") as f:
            sentences = f.read().split("\n")
        
        noised_sentences = [noising(sentence, lambda_value) for sentence in sentences]

        src_file = os.path.basename(file)
        if src_file.startswith("tmp."):
            src_file = src_file.replace("tmp.", "")

        with open(os.path.join(output_dir, f"{src_file}"), "w") as f:
            f.write("\n".join(noised_sentences))
        
    # Copy test/validation data
    for lang in [src_lang, tgt_lang]:
        for s in ["valid", "test"]:
            os.system(f"cp {src_dir}tmp.{s}.{lang} {os.path.join(output_dir, f'{s}.{lang}')}")
    

def noising(sentence: str, lambda_value: float):
    """
    Apply mBART-style noising to a sentence.
    
    Args:
        sentence (str): Input sentence.
        lambda_value (float): Lambda parameter for Poisson distribution.
    
    Returns:
        str: Noised sentence.
    """
    words = sentence.split()

    wl = len(words)

    if wl == 0:
        return ""
    
    mask_length = min(wl, poisson(lam=lambda_value))

    if mask_length == wl:
        mask_start = 0
    else:
        mask_start = random.choice(range(wl - mask_length))

    words = ["<mask>" if i >= mask_start and i < mask_start + mask_length else words[i] for i in range(len(words))]
    # del words[mask_start:mask_start+mask_length]
    # words.insert(mask_start, "_")
   
    return " ".join(words)