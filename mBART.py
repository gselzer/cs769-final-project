from numpy.random import poisson
import random
import os
from typing import List

def mbart(src_files: List[str], tgt_files: List[str], output_dir: str, lambda_value: float = 3.5):
    """
    Apply noising to each sentence in the input file and save the result to the output file.
    
    Args:
        infile (str): Input file path.
        outfile (str): Output file path.
        lambda_value (float): Lambda parameter for Poisson distribution.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(src_files)):

        with open(src_files[i], "r") as f:
            sentences = f.read().split("\n")
        
        noised_sentences = [noising(sentence, lambda_value) for sentence in sentences]

        src_file = os.path.basename(src_files[i])
        if src_file.startswith("tmp."):
            src_file = src_file.replace("tmp.", "")

        with open(os.path.join(output_dir, f"{src_file[i]}"), "w") as f:
            f.write("\n".join(noised_sentences))

        os.system(f"cp {src_files[i]} {tgt_files[i]}")

def noising(sentence: str, lambda_value):
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

    words[mask_start:mask_start+mask_length] = "_" * mask_length
    # del words[mask_start:mask_start+mask_length]
    # words.insert(mask_start, "_")
   
    return " ".join(words)