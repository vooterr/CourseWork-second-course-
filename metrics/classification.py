from typing import List
from collections import Counter


def classification_accuracy(preds: List[str], labels: List[str]) -> float:
    correct = sum(p == y for p, y in zip(preds, labels))
    return correct / len(labels)


def macro_f1(preds: List[str], labels: List[str]) -> float:
    classes = set(labels)
    f1s = []

    for cls in classes:
        tp = sum((p == cls) and (y == cls) for p, y in zip(preds, labels))
        fp = sum((p == cls) and (y != cls) for p, y in zip(preds, labels))
        fn = sum((p != cls) and (y == cls) for p, y in zip(preds, labels))

        if tp == 0:
            f1s.append(0.0)
            continue

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))

    return sum(f1s) / len(f1s)
