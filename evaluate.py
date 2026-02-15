"""
evaluate.py
===========
Evaluation metrics and helpers for the classifier pipeline.
"""

from collections import Counter, defaultdict


def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true) if y_true else 0.0


def precision_recall_f1(y_true, y_pred, pos_label):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == pos_label and b == pos_label)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != pos_label and b == pos_label)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == pos_label and b != pos_label)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def confusion_matrix_2x2(y_true, y_pred, classes):
    """Returns dict  {(actual, predicted): count}."""
    cm = defaultdict(int)
    for a, b in zip(y_true, y_pred):
        cm[(a, b)] += 1
    return cm


def run_all_experiments(train_texts, train_labels, test_texts, test_labels,
                        feature_extractors, classifiers):
    """
    Train every (feature, classifier) combination.

    Returns a list of result dicts with keys:
        feature, classifier, train_acc, test_acc, per_class, cm, classes,
        model, vectoriser, test_preds, test_labels
    """
    classes = sorted(set(train_labels))
    results = []

    for feat_name, feat_factory in feature_extractors.items():
        vec = feat_factory()
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)

        for clf_name, clf_factory in classifiers.items():
            clf = clf_factory()
            clf.fit(X_train, train_labels)

            train_preds = clf.predict(X_train)
            test_preds = clf.predict(X_test)

            train_acc = accuracy(train_labels, train_preds)
            test_acc = accuracy(test_labels, test_preds)

            per_class = {}
            for c in classes:
                p, r, f = precision_recall_f1(test_labels, test_preds, c)
                per_class[c] = {"precision": p, "recall": r, "f1": f}

            cm = confusion_matrix_2x2(test_labels, test_preds, classes)

            results.append({
                "feature": feat_name,
                "classifier": clf_name,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "per_class": per_class,
                "cm": cm,
                "classes": classes,
                "model": clf,
                "vectoriser": vec,
                "test_preds": test_preds,
                "test_labels": test_labels,
            })

    return results


def print_comparison_table(results):
    classes = results[0]["classes"] if results else []
    header = f"{'Feature':<10} {'Classifier':<22} {'Train Acc':>10} {'Test Acc':>10} "
    for c in classes:
        header += f"  P({c[:5]}) R({c[:5]}) F1({c[:5]})"
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r['feature']:<10} {r['classifier']:<22} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f} "
        for c in r["classes"]:
            m = r["per_class"][c]
            row += f"  {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>8.4f}"
        print(row)


def print_confusion_matrix(cm, classes):
    header = f"{'':>12} | " + " | ".join(f"{c:>12}" for c in classes) + " |"
    sep = "-" * len(header)
    print(sep)
    print(f"{'Actual ↓ / Pred →':>12}")
    print(header)
    print(sep)
    for actual in classes:
        row = f"{actual:>12} | "
        row += " | ".join(f"{cm.get((actual, pred), 0):>12}" for pred in classes)
        row += " |"
        print(row)
    print(sep)
