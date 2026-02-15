"""
main.py
=======
Entry point for the Sport vs Politics NLP Classifier.

Usage
-----
    python main.py              # interactive menu
    python main.py --auto       # train + evaluate + generate charts non-interactively
"""

import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

from load_data import load_dataset, prepare_splits
from preprocess import FEATURE_EXTRACTORS, tokenize
from train_models import CLASSIFIERS
from evaluate import (
    run_all_experiments,
    print_comparison_table,
    print_confusion_matrix,
)
from visualize import generate_all_charts


# ─────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────

class Logger:
    def __init__(self, path):
        self.path = path
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("=== Sport vs Politics Classification Log ===\n")
            f.write(f"timestamp={datetime.now().isoformat(timespec='seconds')}\n---\n")

    def section(self, title):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"\n[{title}] timestamp={datetime.now().isoformat(timespec='seconds')}\n")

    def write(self, text):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ─────────────────────────────────────────────
# Auto mode  (non-interactive)
# ─────────────────────────────────────────────

def auto_mode():
    base = Path(__file__).resolve().parent
    logger = Logger(base / "run.log")

    print("Loading dataset …")
    sport_data, politics_data = load_dataset(base)
    print(f"  {len(sport_data)} sport  +  {len(politics_data)} politics sentences")

    train_texts, train_labels, test_texts, test_labels = prepare_splits(
        sport_data, politics_data
    )
    all_labels = [l for _, l in sport_data + politics_data]

    print(f"  Train: {len(train_texts)}   Test: {len(test_texts)}\n")

    print("Training 9 model combinations (3 features × 3 classifiers) …")
    results = run_all_experiments(
        train_texts, train_labels,
        test_texts, test_labels,
        FEATURE_EXTRACTORS, CLASSIFIERS,
    )

    # Log results
    for r in results:
        logger.section(f"{r['feature']} + {r['classifier']}")
        logger.write(f"train_accuracy={r['train_acc']:.6f}")
        logger.write(f"test_accuracy={r['test_acc']:.6f}")
        for c in r["classes"]:
            m = r["per_class"][c]
            logger.write(f"  {c}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    print("\n=== Comparison Table ===")
    print_comparison_table(results)

    best = max(results, key=lambda r: r["test_acc"])
    print(f"\nBest: {best['feature']} + {best['classifier']}  ({best['test_acc']*100:.2f}%)\n")

    print("Generating visualizations …")
    paths = generate_all_charts(results, all_labels)
    for p in paths:
        print(f"  saved → {p}")

    print(f"\nLog saved to: {base / 'run.log'}")
    print("Done.")


# ─────────────────────────────────────────────
# Interactive menu
# ─────────────────────────────────────────────

def interactive_mode():
    base = Path(__file__).resolve().parent
    logger = Logger(base / "run.log")

    sport_data, politics_data = load_dataset(base)
    print(f"Loaded {len(sport_data)} sport + {len(politics_data)} politics sentences.\n")

    train_texts, train_labels, test_texts, test_labels = prepare_splits(
        sport_data, politics_data
    )
    all_labels = [l for _, l in sport_data + politics_data]
    results = None

    while True:
        print("\n" + "=" * 60)
        print("  Sport vs Politics Text Classifier")
        print("=" * 60)
        print("  1 — Dataset statistics")
        print("  2 — Train & evaluate all models")
        print("  3 — Show comparison table")
        print("  4 — Show confusion matrices")
        print("  5 — Predict a custom sentence")
        print("  6 — Generate visualization charts")
        print("  7 — Reload dataset")
        print("  q — Quit")
        print("=" * 60)

        choice = input("Enter option: ").strip().lower()

        if choice == "1":
            all_data = sport_data + politics_data
            label_dist = Counter(l for _, l in all_data)
            avg_len = sum(len(tokenize(t, remove_stopwords=False)) for t, _ in all_data) / len(all_data)
            print(f"\nTotal samples       : {len(all_data)}")
            print(f"Training samples    : {len(train_texts)}")
            print(f"Test samples        : {len(test_texts)}")
            print("Class distribution  :")
            for label, count in sorted(label_dist.items()):
                print(f"  {label}: {count}  ({count / len(all_data) * 100:.1f}%)")
            print(f"Avg tokens/sentence : {avg_len:.1f}")

        elif choice == "2":
            print("\nTraining 9 model combinations (3 features × 3 classifiers) …")
            results = run_all_experiments(
                train_texts, train_labels,
                test_texts, test_labels,
                FEATURE_EXTRACTORS, CLASSIFIERS,
            )
            for r in results:
                logger.section(f"{r['feature']} + {r['classifier']}")
                logger.write(f"train_accuracy={r['train_acc']:.6f}")
                logger.write(f"test_accuracy={r['test_acc']:.6f}")
                for c in r["classes"]:
                    m = r["per_class"][c]
                    logger.write(f"  {c}: P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
            best = max(results, key=lambda r: r["test_acc"])
            print(f"Done.  Best: {best['feature']} + {best['classifier']}  "
                  f"({best['test_acc']*100:.2f}%)")

        elif choice == "3":
            if not results:
                print("Train first (option 2).")
            else:
                print_comparison_table(results)

        elif choice == "4":
            if not results:
                print("Train first (option 2).")
            else:
                for r in results:
                    print(f"\n--- {r['feature']} + {r['classifier']} ---")
                    print_confusion_matrix(r["cm"], r["classes"])

        elif choice == "5":
            if not results:
                print("Train first (option 2).")
                continue
            print("\nAvailable models:")
            for i, r in enumerate(results):
                print(f"  {i+1}. {r['feature']} + {r['classifier']}  "
                      f"(test acc {r['test_acc']*100:.2f}%)")
            print("  0. All models")
            mc = input("Choose model number: ").strip()
            sentence = input("Enter sentence: ").strip()
            if not sentence:
                print("Empty input — skipped.")
                continue
            chosen = results if mc == "0" else []
            if mc != "0":
                try:
                    chosen = [results[int(mc) - 1]]
                except (ValueError, IndexError):
                    print("Invalid choice — using all models.")
                    chosen = results
            for r in chosen:
                vec = r["vectoriser"].transform([sentence])
                pred = r["model"].predict_one(vec[0])
                proba = r["model"].predict_proba_one(vec[0])
                proba_str = " | ".join(f"{c}={proba.get(c,0):.4f}" for c in sorted(proba))
                print(f"  [{r['feature']} + {r['classifier']}]  → {pred}   ({proba_str})")

        elif choice == "6":
            if not results:
                print("Train first (option 2).")
            else:
                print("Generating charts …")
                paths = generate_all_charts(results, all_labels)
                for p in paths:
                    print(f"  saved → {p}")

        elif choice == "7":
            sport_data, politics_data = load_dataset(base)
            train_texts, train_labels, test_texts, test_labels = prepare_splits(
                sport_data, politics_data
            )
            all_labels = [l for _, l in sport_data + politics_data]
            results = None
            print(f"Reloaded {len(sport_data)} sport + {len(politics_data)} politics sentences.")

        elif choice in {"q", "quit"}:
            print("Bye.")
            break
        else:
            print("Invalid option.")

    print(f"\nLog → {base / 'run.log'}")


# ─────────────────────────────────────────────

if __name__ == "__main__":
    if "--auto" in sys.argv:
        auto_mode()
    else:
        interactive_mode()
