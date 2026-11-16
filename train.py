from src.data import load_and_preprocess
from src.model import train

if __name__ == "__main__":
    frame = load_and_preprocess("data/german.data")
    _, results = train(frame)
    for key, value in results.items():
        if key in {"roc_curve", "test_targets", "test_probabilities",
                   "reliability_bins", "purpose_error_analysis"}:
            continue
        if key == "baseline":
            value = {name: item for name, item in value.items()
                     if name not in {"test_probabilities", "reliability_bins"}}
        elif key == "training_prevalence_baseline":
            value = {name: item for name, item in value.items()
                     if name != "reliability_bins"}
        print(f"{key}: {value}")

