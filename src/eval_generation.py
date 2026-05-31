"""Automated generation quality evaluation for Bonus diffusion LMs.

Computes:
  - Type-Token Ratio (lexical diversity)
  - Repetition rate (n-gram overlap)
  - Average sentence length
  - Perplexity under an external GPT evaluator (if available)

Usage:
    python eval_generation.py --bonus_dir ../results/bonus
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Evaluate generation quality')
parser.add_argument('--bonus_dir', type=str, default='../results/bonus')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BONUS_DIR = os.path.join(SCRIPT_DIR, args.bonus_dir)


def load_samples(json_path):
    with open(json_path) as f:
        return json.load(f).get('samples', [])


def compute_metrics(samples):
    """Compute diversity and quality metrics for a list of generated texts."""
    if not samples:
        return {}

    texts = [s.strip() for s in samples if s.strip()]
    if not texts:
        return {}

    all_tokens = []
    doc_tokens = []
    for text in texts:
        tokens = text.lower().split()
        doc_tokens.append(tokens)
        all_tokens.extend(tokens)

    # Type-Token Ratio (overall lexical diversity)
    ttr = len(set(all_tokens)) / max(len(all_tokens), 1)

    # Mean TTR per sample (per-document diversity)
    per_doc_ttr = [len(set(t)) / max(len(t), 1) for t in doc_tokens if t]
    mean_ttr = sum(per_doc_ttr) / max(len(per_doc_ttr), 1)

    # Self-BLEU / repetition rate (bigram overlap between pairs)
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    total_bigram_overlap = 0
    n_pairs = 0
    for i in range(len(doc_tokens)):
        for j in range(i + 1, len(doc_tokens)):
            bi = ngrams(doc_tokens[i], 2)
            bj = ngrams(doc_tokens[j], 2)
            intersection = sum((bi & bj).values())
            union = max(sum((bi | bj).values()), 1)
            total_bigram_overlap += intersection / union
            n_pairs += 1
    mean_pairwise_overlap = total_bigram_overlap / max(n_pairs, 1)
    mean_len = sum(len(t) for t in doc_tokens) / len(doc_tokens)

    # Truncated repetition detection
    def has_repetition(text, min_ngram=3, min_repeat=3):
        words = text.lower().split()
        for n in range(min_ngram, min(6, len(words) // 2)):
            for i in range(len(words) - n * min_repeat):
                gram = tuple(words[i:i + n])
                count = sum(1 for j in range(len(words) - n + 1)
                            if tuple(words[j:j + n]) == gram)
                if count >= min_repeat:
                    return True
        return False

    n_repetitive = sum(1 for t in texts if has_repetition(t))
    repetition_rate = n_repetitive / len(texts)

    return {
        'n_samples': len(texts),
        'ttr_overall': round(ttr, 4),
        'ttr_mean_per_doc': round(mean_ttr, 4),
        'mean_length_tokens': round(mean_len, 1),
        'pairwise_bigram_overlap': round(mean_pairwise_overlap, 4),
        'repetition_rate': round(repetition_rate, 4),
    }


def main():
    results = {}
    for model_name in ['diffusion', 'llada']:
        model_metrics = {}
        for size in ['d64', 'd128', 'd192']:
            path = os.path.join(BONUS_DIR, f'results_{model_name}_{size}.json')
            if not os.path.exists(path):
                continue
            samples = load_samples(path)
            metrics = compute_metrics(samples)
            model_metrics[size] = metrics
            print(f"\n{model_name} {size}:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        results[model_name] = model_metrics

    print(f"\n{'='*60}")
    print("Comparison at M size (d128)")
    for model_name in ['diffusion', 'llada']:
        m = results.get(model_name, {}).get('d128', {})
        if m:
            print(f"\n{model_name}:")
            print(f"  TTR (overall):     {m.get('ttr_overall', 'N/A')}")
            print(f"  Mean doc TTR:      {m.get('ttr_mean_per_doc', 'N/A')}")
            print(f"  Repetition rate:   {m.get('repetition_rate', 'N/A')}")
            print(f"  Bigram overlap:    {m.get('pairwise_bigram_overlap', 'N/A')}")

    with open(os.path.join(BONUS_DIR, 'generation_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {os.path.join(BONUS_DIR, 'generation_metrics.json')}")

if __name__ == '__main__':
    main()
