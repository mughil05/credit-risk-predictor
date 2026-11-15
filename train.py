from src.data import load_and_preprocess
from src.model import train

if __name__ == "__main__":
    frame = load_and_preprocess("data/german.data")
    _, results = train(frame)
    for key, value in results.items():
        if key not in {"roc_curve", "test_targets", "test_probabilities"}:
            print(f"{key}: {value}")

