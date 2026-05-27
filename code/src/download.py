# coding: utf-8
"""Download a domain-specific dataset for Part C fine-tuning.

Default: Chinese Wikipedia (encyclopedic domain).
The dataset is saved as plain text files compatible with the PTB format
used in Parts A/B: one sentence per line, whitespace tokenization optional.

Usage:
    python download.py                          # default: Chinese Wikipedia
    python download.py --dataset wikitext       # English Wikipedia (wikitext-2)
    python download.py --dataset arxiv          # CS/ML abstracts
    python download.py --dataset custom --custom_file /path/to/text.txt
"""

import argparse
import json
import os
import re
import sys

from datasets import load_dataset, get_dataset_config_names

parser = argparse.ArgumentParser(description='Download domain dataset for Part C')
parser.add_argument('--dataset', type=str, default='wikipedia-zh',
                    choices=['wikipedia-zh', 'wikipedia-en', 'arxiv',
                             'shakespeare', 'poetry-zh', 'ted-talks',
                             'fairy-tales', 'custom'],
                    help='which dataset to download')
parser.add_argument('--custom_file', type=str, default=None,
                    help='path to custom text file (for --dataset custom)')
parser.add_argument('--output_dir', type=str, default='../data/domain',
                    help='output directory for train.txt and valid.txt')
parser.add_argument('--max_lines', type=int, default=50000,
                    help='maximum lines to save (to keep dataset small)')
parser.add_argument('--valid_ratio', type=float, default=0.1,
                    help='fraction of data for validation')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, args.output_dir)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    """Basic cleaning: strip, collapse whitespace, remove empty lines."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def save_splits(lines, train_path, valid_path, valid_ratio):
    """Shuffle lines and split into train/valid."""
    import random
    random.seed(42)
    random.shuffle(lines)

    n_valid = max(1, int(len(lines) * valid_ratio))
    valid_lines = lines[:n_valid]
    train_lines = lines[n_valid:]

    for path, subset in [(train_path, train_lines), (valid_path, valid_lines)]:
        with open(path, 'w', encoding='utf-8') as f:
            for line in subset:
                f.write(line + '\n')
        size_kb = os.path.getsize(path) / 1024
        print(f"  Saved {len(subset):,} lines ({size_kb:.1f} KB) → {path}")


# ─── Dataset-specific download logic ──────────────────────────────────────────

if args.dataset == 'custom':
    if not args.custom_file or not os.path.exists(args.custom_file):
        print("Please provide --custom_file pointing to an existing text file.")
        sys.exit(1)
    with open(args.custom_file, 'r', encoding='utf-8') as f:
        lines = [clean_text(line) for line in f if clean_text(line)]
    lines = lines[:args.max_lines]

elif args.dataset == 'wikipedia-zh':
    print("Downloading Chinese Wikipedia (20220301.zh) ...")
    dataset = load_dataset("wikipedia", "20220301.zh", split="train", streaming=True)
    lines = []
    for item in dataset:
        text = clean_text(item['text'])
        # Split long articles into individual sentences/paragraphs
        for para in text.split('\n'):
            para = clean_text(para)
            if len(para) > 20:  # skip very short fragments
                lines.append(para)
        if len(lines) >= args.max_lines:
            break
    lines = lines[:args.max_lines]

elif args.dataset == 'wikipedia-en':
    print("Downloading English Wikipedia (20220301.en) ...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    lines = []
    for item in dataset:
        text = clean_text(item['text'])
        for para in text.split('\n'):
            para = clean_text(para)
            if len(para) > 30:
                lines.append(para)
        if len(lines) >= args.max_lines:
            break
    lines = lines[:args.max_lines]

elif args.dataset == 'arxiv':
    print("Downloading arXiv abstracts (scientific writing style) ...")
    arxiv_sources = [
        ("p208p2002/arxiv-abstracts", "abstract"),
        ("gfissore/arxiv-abs", "abstract"),
        ("ccdv/arxiv-classification", "text"),
    ]
    lines = []
    for ds_id, text_field in arxiv_sources:
        try:
            ds = load_dataset(ds_id, split="train", streaming=True)
            for item in ds:
                text = clean_text(item.get(text_field, ''))
                if len(text) > 60:
                    lines.append(text)
                if len(lines) >= args.max_lines:
                    break
            if lines:
                print(f"  Loaded from: {ds_id}")
                break
        except Exception:
            continue
    if not lines:
        print("  All arXiv sources failed. Falling back to English Wikipedia.")
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        for item in ds:
            text = clean_text(item['text'])
            for para in text.split('\n'):
                para = clean_text(para)
                if len(para) > 30:
                    lines.append(para)
            if len(lines) >= args.max_lines:
                break
    lines = lines[:args.max_lines]

elif args.dataset == 'shakespeare':
    print("Downloading Shakespeare (Elizabethan English style) ...")
    ds = load_dataset("tiny_shakespeare", split="train")
    full_text = " ".join(ds["text"])
    sentences = re.split(r'(?<=[.!?:;])\s+', full_text)
    lines = []
    for sent in sentences:
        sent = clean_text(sent)
        if 30 < len(sent) < 300:
            lines.append(sent)
        if len(lines) >= args.max_lines:
            break
    lines = lines[:args.max_lines]

elif args.dataset == 'poetry-zh':
    print("Downloading Chinese classical poetry ...")
    try:
        ds = load_dataset("shuishu/chinese_poetry_collection", split="train", streaming=True)
        lines = []
        for item in ds:
            paragraphs = item.get('paragraphs', [])
            for para in paragraphs:
                para = clean_text(para)
                if len(para) > 5:
                    lines.append(para)
            if len(lines) >= args.max_lines:
                break
        lines = lines[:args.max_lines]
    except Exception:
        print("  HF dataset unavailable. Downloading from chinese-poetry GitHub...")
        import urllib.request
        url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/json/poet.tang.0.json"
        try:
            data = json.loads(urllib.request.urlopen(url, timeout=30).read())
            lines = []
            for poem in data:
                for para in poem.get('paragraphs', []):
                    para = clean_text(para)
                    if len(para) > 5:
                        lines.append(para)
                if len(lines) >= args.max_lines:
                    break
            lines = lines[:args.max_lines]
        except Exception:
            print("  Download failed. Try --dataset custom with your own text.")
            sys.exit(1)

elif args.dataset == 'ted-talks':
    print("Downloading TED talks (presentation/speech style) ...")
    try:
        ds = load_dataset("gigant/ted_talks_iwslt", "en", split="train", streaming=True)
        lines = []
        for item in ds:
            text = clean_text(item.get('text', item.get('talk', '')))
            if len(text) > 40:
                lines.append(text)
            if len(lines) >= args.max_lines:
                break
        lines = lines[:args.max_lines]
    except Exception:
        print("  TED unavailable. Trying OpenSubtitles (spoken dialogue style)...")
        try:
            ds = load_dataset("opensubtitles", split="train", streaming=True)
            lines = []
            for item in ds:
                text = clean_text(item.get('text', ''))
                if len(text) > 30:
                    lines.append(text)
                if len(lines) >= args.max_lines:
                    break
            lines = lines[:args.max_lines]
        except Exception:
            print("  All fallbacks failed. Try --dataset custom.")
            sys.exit(1)

elif args.dataset == 'fairy-tales':
    print("Downloading fairy tales / storytelling style ...")
    try:
        ds = load_dataset("copenlu/fairy_tales_qa", split="train", streaming=True)
        lines = []
        for item in ds:
            text = clean_text(item.get('story', item.get('text', '')))
            if len(text) > 40:
                lines.append(text)
            if len(lines) >= args.max_lines:
                break
        lines = lines[:args.max_lines]
    except Exception:
        print("  Fairy tales unavailable. Trying BookCorpus (narrative style)...")
        try:
            ds = load_dataset("bookcorpus", split="train", streaming=True)
            lines = []
            for item in ds:
                text = clean_text(item['text'])
                for sent in re.split(r'(?<=[.!?])\s+', text):
                    sent = clean_text(sent)
                    if 30 < len(sent) < 500:
                        lines.append(sent)
                if len(lines) >= args.max_lines:
                    break
            lines = lines[:args.max_lines]
        except Exception:
            print("  All fallbacks failed. Try --dataset custom.")
            sys.exit(1)

# ─── Save ─────────────────────────────────────────────────────────────────────

train_path = os.path.join(OUTPUT_DIR, 'train.txt')
valid_path = os.path.join(OUTPUT_DIR, 'valid.txt')
save_splits(lines, train_path, valid_path, args.valid_ratio)

print(f"\nDataset ready: {OUTPUT_DIR}/")
print(f"  train.txt: {os.path.getsize(train_path)/1024:.1f} KB")
print(f"  valid.txt: {os.path.getsize(valid_path)/1024:.1f} KB")
