"""train_detector.py — Train fault detector on episode JSONL or synthetic data."""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np

CHECKPOINT = Path("results/checkpoints/detector.pkl")

def load_jsonl(path: str):
    X, y = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            feat = rec.get("features", [])
            label = rec.get("fault_label", 0)
            if feat:
                X.append(feat)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

def train(X, y, model_path: Path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    clf.fit(X_tr, y_tr)
    report = classification_report(y_val, clf.predict(X_val), zero_division=0)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Detector trained. Report:\n{report}")
    print(f"Saved to {model_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="JSONL episode dataset")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-path", default=str(CHECKPOINT))
    args = ap.parse_args()
    mp = Path(args.model_path)
    if args.data and Path(args.data).exists():
        X, y = load_jsonl(args.data)
        print(f"Loaded {len(X)} samples from {args.data}")
    else:
        # Synthetic
        from src.detector import generate_synthetic_detector_data
        X, y = generate_synthetic_detector_data(n_samples=args.samples, seed=args.seed)
        print(f"Generated {len(X)} synthetic samples")
    train(X, y, mp)

if __name__ == "__main__":
    main()

