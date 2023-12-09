import argparse
import shutil
import compare_mt.compare_mt_main
import re
import os
from typing import List

def compare(args: argparse.Namespace):
    # Validate paths
    model1_dir = args.m1
    model2_dir = args.m2
    if not os.path.exists(model1_dir):
        print(f"Path {model1_dir} does not exist")
    if not os.path.exists(model2_dir):
        print(f"Path {model2_dir} does not exist")
    
    # Grab output files
    out1 = os.path.join(model1_dir, "generate-test.txt")
    out2 = os.path.join(model2_dir, "generate-test.txt")
    if os.path.exists(out1):
        with open(out1, "r") as f:
            lines1  = f.readlines()
    else:
        raise Exception(f"Model output file {out1} does not exist")
    if os.path.exists(out2):
        with open(out2, "r") as f:
            lines2  = f.readlines()
    else:
        raise Exception(f"Model output file {out2} does not exist")
    
    # Parse the targets
    expected = parse_targets(lines1)

    # Parse both model test files
    actual1 = parse_outputs(lines1)
    actual2 = parse_outputs(lines2)
    
    # Write out targets
    base1 = os.path.basename(model1_dir)
    base2 = os.path.basename(model2_dir)
    out_dir = os.path.join("comparisons", f"{base1}_and_{base2}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    expected_file = os.path.join(out_dir, "expected")
    with open(expected_file, mode="w") as f:
        f.writelines("\n".join(expected))

    # Write out hypotheses
    actual1_file = os.path.join(out_dir, base1)
    actual2_file = os.path.join(out_dir, base2)
    with open(actual1_file, mode="w") as f:
        f.writelines("\n".join(actual1))
    with open(actual2_file, mode="w") as f:
        f.writelines("\n".join(actual2))

    # Comparison with significance testing
    if args.st:
        os.system(f"compare-mt --output_directory {out_dir}/results {expected_file} {actual1_file} {actual2_file} --compare_scores score_type=bleu,bootstrap=1000,prob_thresh=0.05")
        return

    # Write the comparison
    os.system(f"compare-mt --output_directory {out_dir}/results {expected_file} {actual1_file} {actual2_file}")
    return

def parse_targets(lines: List[str]):
    return _parse(lines, r'T-(\d+)\t(.+)\n')

def parse_outputs(lines: List[str]):
    return _parse(lines, r'H-(\d+)\t.+\t(.+)\n')

def _parse(lines: List[str], rgx: str):
    parsed = []
    for line in lines:
        match = re.match(rgx, line)
        if match:
            idx = int(match.group(1))
            sentence = match.group(2)
            parsed.append((idx, sentence))
    
    sorted_tuples = sorted(parsed, key=lambda x: x[0])
    sorted_lines = [i[1] for i in sorted_tuples]
    detokenized = [i.replace("@@ ", "") for i in sorted_lines]
    
    return detokenized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parse source and target languages
    parser.add_argument('--m1', help='the first model output directory', type=os.path.abspath, default="tmp/optimized_de_en")
    parser.add_argument('--m2', help='the second model output directory', type=os.path.abspath, default="tmp/optimized_de_en_pos")
    parser.add_argument('--st', help='significance testing', action=argparse.BooleanOptionalAction)
    compare(parser.parse_args())