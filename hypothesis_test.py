"""
Example command:
    python hypothesis_test.py -b experiments/results/optimized_de_en \
        -c experiments/results/optimized_de_en_data_diversification experiments/results/optimized_de_en_joint-dropout \ 
        -p 0.5 -N 1000
"""

import argparse
import os
import random
import subprocess
import re
from scipy.stats import ttest_rel
from tqdm import tqdm

def score(trg_file_path:str, tran_file_path:str):
    """ Generate sacre bleu score """
    command = f"fairseq-score --sys {tran_file_path} --ref {trg_file_path}"
    result = subprocess.run('fairseq-score --sys tmp/gen.out.sys --ref tmp/gen.out.ref',
                        shell=True, text=True, capture_output=True)
    bleu_score =  None
    if result.returncode == 0:
        match = re.search(r'BLEU4 = ([0-9.]+),', result.stdout)
        if match:
            bleu_score = float(match.group(1))
    else:
        raise Exception("Error in executing command:",result.stderr)

    return bleu_score


def parse_generate_test_file(file_path:str):
    """" Returns a list of target sentences and a list of generated translations """
    
    trg_sentences, translation_sentences = [], []
    
    with open(os.path.join(file_path,'generate-test.txt'),'r') as file:
        for line in file:
            if line.startswith('T-'):
                content_start = line.find('\t') + 1
                trg_sentences.append(line[content_start:-1])
            elif line.startswith('H-'):
                parts = line.strip().split('\t')
                translation_sentences.append(parts[2])
    
    return trg_sentences, translation_sentences


def generate_bootstrap_scores(file_path:str, n_samples:int, N:int):
    """ Generates a list of scores by bootstraping output test file """
    
    trg_sentences, translation_sentences = parse_generate_test_file(file_path)

    dir_path = 'tmp'
    os.makedirs(dir_path, exist_ok=True)
    file_path_trg = os.path.join(dir_path, 'gen.out.ref')
    file_path_tran = os.path.join(dir_path, 'gen.out.sys')

    scores = []
    exp_name = experiment_name(file_path)
    for _ in tqdm(range(N), desc=f"Boostrapping {exp_name}"):
        with open(file_path_trg, 'w') as file_trg, open(file_path_tran, 'w') as file_tran:
            for _ in range(n_samples):
                idx = random.randint(0,n_samples-1)
                file_trg.write(trg_sentences[idx]+'\n')
                file_tran.write(translation_sentences[idx]+'\n') 
            scores.append(score(file_path_trg, file_path_tran))

    return scores


def num_samples(file_path:str):
    """Count the number of source-translation sentance pairs"""

    path = os.path.join(file_path, 'generate-test.txt')
    result = subprocess.run(['wc', '-l', path], capture_output=True, text=True, check=True)

    line_count = int(result.stdout.split()[0])

    return int((line_count - 1) / 5)


def t_test(base_scores:list, comparison_scores:list, alpha:int):
    t_statistic, p_value = ttest_rel(base_scores, comparison_scores)
    
    if p_value < alpha:
        print(f'\t The difference in BLEU scores is statistically significant.\n\t\tp={p_value}')
    else:
        print(f'\t The difference in BLEU scores is not statistically significant.\n\t\tp={p_value}')


def experiment_name(file_path:str):
    path_components = file_path.split(os.sep)

    for component in path_components:
        if component.startswith('optimized_'):
            return component

    return None


def main():
    parser = argparse.ArgumentParser(description="Performs boostrapping and t-test")
    parser.add_argument(
            '-b', '--base-method',  
            default='experiments/results/optimized_de_en',
            help='Path to base method output')
    parser.add_argument(
            '-c', '--comparison-methods', 
            nargs='+', 
            default=['experiments/results/optimized_de_en_data_diversification',
                     'experiments/results/optimized_de_en_joint-dropout'],
            help='Paths to methd outputs to be compared')
    parser.add_argument('-N', type=int, default=1000, help='Number of bootstrap rounds.')
    parser.add_argument('-p','--p-value', type=float, default=0.05, help='Significance level for the t-test (default: 0.05).')

    args = parser.parse_args()

    n_samples = num_samples(args.base_method)
    for path in args.comparison_methods:
        if n_samples != num_samples(path):
            raise Exception(f"Test file in {path} is a different length than the base method file")
    
    print(f"\nGenerating bootstrap scores for base method: {experiment_name(args.base_method)}")
    base_scores = generate_bootstrap_scores(args.base_method, n_samples, args.N)
   
    comparison_scores = []
    for path in args.comparison_methods:
        print(f"\nGenerating bootstrap scores for comparison method: {experiment_name(path)}")
        comparison_scores.append(generate_bootstrap_scores(path, n_samples, args.N))

    base_experiment_name = experiment_name(args.base_method)
    for i in range(len(args.comparison_methods)):
        comparision_experiment_name = experiment_name(args.comparison_methods[i])
        print(f"\nComparing {base_experiment_name} with {comparision_experiment_name}:")
        t_test(base_scores, comparison_scores[i], args.p_value)
    

if __name__ == "__main__":
    main()
