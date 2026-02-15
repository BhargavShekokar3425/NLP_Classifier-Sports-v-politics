"""
interactive.py
==============
Interactive sentence classifier.

Trains all 9 model combinations on startup, then lets you type
sentences repeatedly. Each sentence is classified by every model,
results are displayed in a ranked table, and everything is logged
to ``interactive.log``.

Usage
-----
    python3 interactive.py
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

from load_data import load_dataset, prepare_splits
from preprocess import FEATURE_EXTRACTORS, tokenize
from train_models import CLASSIFIERS
from evaluate import accuracy, precision_recall_f1


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

BASE = Path(__file__).resolve().parent
LOG_PATH = BASE / "interactive.log"


def _log(text, mode="a"):
    with open(LOG_PATH, mode, encoding="utf-8") as f:
        f.write(text + "\n")


def _banner(text, char="═", width=70):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _color(text, code):
    """ANSI color wrapper (works in most terminals)."""
    return f"\033[{code}m{text}\033[0m"


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_all_models():
    """Load data, train all 9 combos, return results list."""
    sport_data, politics_data = load_dataset(BASE)
    train_texts, train_labels, test_texts, test_labels = prepare_splits(
        sport_data, politics_data
    )

    total = len(sport_data) + len(politics_data)
    print(f"  Dataset : {total} sentences  "
          f"({len(sport_data)} sport / {len(politics_data)} politics)")
    print(f"  Split   : {len(train_texts)} train / {len(test_texts)} test")

    results = []
    for feat_name, feat_factory in FEATURE_EXTRACTORS.items():
        vec = feat_factory()
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)

        for clf_name, clf_factory in CLASSIFIERS.items():
            clf = clf_factory()
            clf.fit(X_train, train_labels)

            test_preds = clf.predict(X_test)
            test_acc = accuracy(test_labels, test_preds)

            per_class = {}
            for c in sorted(set(test_labels)):
                p, r, f = precision_recall_f1(test_labels, test_preds, c)
                per_class[c] = {"precision": p, "recall": r, "f1": f}

            results.append({
                "feature": feat_name,
                "classifier": clf_name,
                "test_acc": test_acc,
                "per_class": per_class,
                "model": clf,
                "vectoriser": vec,
            })

    return results, test_labels


# ─────────────────────────────────────────────
# Prediction + display
# ─────────────────────────────────────────────

def predict_sentence(sentence, results):
    """
    Run *sentence* through all trained models.
    Returns a list of dicts sorted by confidence (highest first).
    """
    predictions = []
    for r in results:
        vec = r["vectoriser"].transform([sentence])[0]
        pred = r["model"].predict_one(vec)
        proba = r["model"].predict_proba_one(vec)
        confidence = max(proba.values())
        predictions.append({
            "feature": r["feature"],
            "classifier": r["classifier"],
            "prediction": pred,
            "confidence": confidence,
            "proba": proba,
            "test_acc": r["test_acc"],
        })
    # Sort by confidence descending
    predictions.sort(key=lambda p: p["confidence"], reverse=True)
    return predictions


def display_predictions(sentence, predictions):
    """Pretty-print prediction table to terminal."""
    _banner(f"Input: \"{sentence}\"", char="─")

    # Header
    print(f"\n {'#':<4} {'Feature':<10} {'Classifier':<22} "
          f"{'Prediction':<12} {'Confidence':>10}   {'POLITICS':>9} {'SPORT':>9}   {'TestAcc':>8}")
    print("─" * 100)

    for i, p in enumerate(predictions):
        rank = i + 1
        pred_color = "92" if p["prediction"] == "SPORT" else "94"  # green / blue
        pred_str = _color(p["prediction"], pred_color)
        pol = p["proba"].get("POLITICS", 0)
        spo = p["proba"].get("SPORT", 0)

        # Highlight the best row
        prefix = _color("★", "93") if rank == 1 else " "

        print(f" {prefix}{rank:<3} {p['feature']:<10} {p['classifier']:<22} "
              f"{pred_str:<21} {p['confidence']:>9.4f}   "
              f"{pol:>9.4f} {spo:>9.4f}   {p['test_acc']*100:>7.2f}%")

    print("─" * 100)

    # Majority vote
    votes = Counter(p["prediction"] for p in predictions)
    majority = votes.most_common(1)[0]
    best = predictions[0]

    print(f"\n  Majority vote  : {_color(majority[0], '1')}  "
          f"({majority[1]}/{len(predictions)} models)")
    print(f"  Most confident : {best['feature']} + {best['classifier']}  →  "
          f"{_color(best['prediction'], '1')}  "
          f"(conf {best['confidence']:.4f}, test acc {best['test_acc']*100:.2f}%)")


def log_predictions(sentence, predictions):
    """Append prediction details to interactive.log."""
    ts = datetime.now().isoformat(timespec="seconds")
    _log(f"\n[PREDICTION] timestamp={ts}")
    _log(f"  sentence = {sentence}")

    for i, p in enumerate(predictions):
        pol = p["proba"].get("POLITICS", 0)
        spo = p["proba"].get("SPORT", 0)
        _log(f"  {i+1:>2}. {p['feature']:<8} + {p['classifier']:<20}  "
             f"→ {p['prediction']:<10}  conf={p['confidence']:.6f}  "
             f"P(POL)={pol:.6f}  P(SPO)={spo:.6f}  test_acc={p['test_acc']:.4f}")

    votes = Counter(p["prediction"] for p in predictions)
    majority = votes.most_common(1)[0]
    best = predictions[0]
    _log(f"  majority_vote = {majority[0]}  ({majority[1]}/{len(predictions)})")
    _log(f"  best_model    = {best['feature']}+{best['classifier']}  "
         f"conf={best['confidence']:.6f}")


# ─────────────────────────────────────────────
# Model summary table
# ─────────────────────────────────────────────

def display_model_summary(results):
    """Print a table of all trained models and their test accuracy."""
    _banner("Trained Models", char="─")
    print(f"\n {'#':<4} {'Feature':<10} {'Classifier':<22} {'Test Acc':>10}")
    print("─" * 50)
    sorted_r = sorted(results, key=lambda r: r["test_acc"], reverse=True)
    for i, r in enumerate(sorted_r):
        marker = _color("★", "93") if i == 0 else " "
        print(f" {marker}{i+1:<3} {r['feature']:<10} {r['classifier']:<22} "
              f"{r['test_acc']*100:>9.2f}%")
    print("─" * 50)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    # Init log
    _log(f"=== Interactive Classifier Log ===\nstarted={datetime.now().isoformat(timespec='seconds')}\n---",
         mode="w")

    _banner("Sport vs Politics — Interactive Classifier")
    print("\n  Training all 9 models (3 features × 3 classifiers) …\n")

    results, _ = train_all_models()

    print(f"\n  ✓ All {len(results)} models trained.\n")

    _log(f"models_trained={len(results)}")
    for r in results:
        _log(f"  {r['feature']}+{r['classifier']}  test_acc={r['test_acc']:.6f}")

    display_model_summary(results)

    # Prediction history
    history = []

    print("\nCommands:")
    print("  • Type a sentence to classify it")
    print("  • 'models'   — show model summary")
    print("  • 'history'  — show prediction history")
    print("  • 'clear'    — clear history")
    print("  • 'quit'     — exit\n")

    while True:
        try:
            user_input = input(_color("Enter sentence → ", "96")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in {"q", "quit", "exit"}:
            print("Bye.")
            break

        elif cmd == "models":
            display_model_summary(results)
            continue

        elif cmd == "history":
            if not history:
                print("  No predictions yet.")
            else:
                print(f"\n  {'#':<4} {'Majority':<12} {'Best Model':<30} {'Conf':>8}   Sentence")
                print("  " + "─" * 90)
                for i, h in enumerate(history):
                    print(f"  {i+1:<4} {h['majority']:<12} "
                          f"{h['best_feature']}+{h['best_classifier']:<18} "
                          f"{h['best_conf']:>7.4f}   {h['sentence'][:50]}")
            continue

        elif cmd == "clear":
            history.clear()
            print("  History cleared.")
            continue

        # ── Classify the sentence ──
        sentence = user_input
        predictions = predict_sentence(sentence, results)
        display_predictions(sentence, predictions)
        log_predictions(sentence, predictions)

        # Save to history
        votes = Counter(p["prediction"] for p in predictions)
        majority = votes.most_common(1)[0][0]
        best = predictions[0]
        history.append({
            "sentence": sentence,
            "majority": majority,
            "best_feature": best["feature"],
            "best_classifier": best["classifier"],
            "best_conf": best["confidence"],
        })

    # Final log summary
    _log(f"\n[SESSION_END] timestamp={datetime.now().isoformat(timespec='seconds')}")
    _log(f"total_predictions={len(history)}")
    for i, h in enumerate(history):
        _log(f"  {i+1}. majority={h['majority']}  best={h['best_feature']}+{h['best_classifier']}  "
             f"conf={h['best_conf']:.4f}  sentence={h['sentence']}")

    print(f"\nLog saved → {LOG_PATH}")


if __name__ == "__main__":
    main()
