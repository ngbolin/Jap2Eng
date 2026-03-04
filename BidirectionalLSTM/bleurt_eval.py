"""
Usage:
    bleurt_eval.py --output-file=<file> --test-file=<file> --checkpoint=<file> --result-file=<file>

Options:
    --output-file=<file>                    prediction file
    --test-file=<file>                      output file
    --checkpoint=<file>                     BLEURT-20 checkpoint file
    --result-file=<file>                    result file
"""

import os
from bleurt import score
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from docopt import docopt

args = docopt(__doc__)

# Paths
def main():
    cand_path = args['--output-file']
    ref_path = args['--test-file']
    checkpoint = args['--checkpoint']
    res_path = args['--result-file']

    # Load data
    with open(cand_path, 'r', encoding='utf-8') as f:
        candidates = [line.strip() for line in f]
    with open(ref_path, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]

    print(f"Candidates found: {len(candidates)}")
    print(f"References found: {len(references)}")

    # Force Alignment (Emergency Trim)
    if len(candidates) != len(references):
        print("!! Mismatch detected. Trimming to the shortest length to force evaluation...")
        min_len = min(len(candidates), len(references))
        candidates = candidates[:min_len]
        references = references[:min_len]

    # Run Scorer
    print("Loading BLEURT-20 (this takes ~30 seconds)...")
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=candidates)

    # Save results to results file
    with open(res_path, 'w') as f:
        for s in scores:
            f.write(f"{s}\n")

    print(f"Done! Mean BLEURT: {sum(scores)/len(scores):.4f}")
    print(f"Results saved to: {res_path}")

if __name__ == "__main__":
    main()
