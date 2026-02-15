"""
load_data.py
============
Handles loading the sport.txt / politics.txt dataset and
splitting it into train/test sets.
"""

import random
from pathlib import Path


def read_labeled_sentences(file_path, label):
    """Read a text file where each non-empty line is one sample."""
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append((line, label))
    return sentences


def load_dataset(base_path=None):
    """
    Load sport.txt and politics.txt from *base_path*.

    Returns
    -------
    sport_data : list of (text, "SPORT")
    politics_data : list of (text, "POLITICS")
    """
    if base_path is None:
        base_path = Path(__file__).resolve().parent

    base_path = Path(base_path)
    sport_file = base_path / "sport.txt"
    politics_file = base_path / "politics.txt"

    if not sport_file.exists() or not politics_file.exists():
        raise FileNotFoundError(
            f"sport.txt and politics.txt must be present in {base_path}"
        )

    sport_data = read_labeled_sentences(sport_file, "SPORT")
    politics_data = read_labeled_sentences(politics_file, "POLITICS")
    return sport_data, politics_data


def train_test_split(data, train_ratio=0.8, seed=42):
    """Shuffle and split data into train / test lists."""
    data_copy = list(data)
    random.seed(seed)
    random.shuffle(data_copy)
    split = int(len(data_copy) * train_ratio)
    split = max(1, min(split, len(data_copy) - 1))
    return data_copy[:split], data_copy[split:]


def prepare_splits(sport_data, politics_data, train_ratio=0.8, seed=42):
    """
    Merge, shuffle, split, and return texts + labels separately.

    Returns
    -------
    train_texts, train_labels, test_texts, test_labels
    """
    all_data = sport_data + politics_data
    random.seed(seed)
    random.shuffle(all_data)

    train_data, test_data = train_test_split(all_data, train_ratio, seed)

    train_texts = [t for t, _ in train_data]
    train_labels = [l for _, l in train_data]
    test_texts = [t for t, _ in test_data]
    test_labels = [l for _, l in test_data]

    return train_texts, train_labels, test_texts, test_labels
