# prepare_beer_all_aspects.py

import os
import json
import gzip

BASE_DATA = "/media/dinesh/4TBHDD/Misc_Downloads_25102025/Programs_Documents/SER_Metric/noise_injection_beer/data_preprocessing/beer_review"

ASPECTS = {
    0: "appearance",
    1: "aroma",
    2: "palate"
}

neg_thres = 0.4
pos_thres = 0.6

ANNOTATION_FILE = os.path.join(BASE_DATA, "annotations.json")

for aspect, name in ASPECTS.items():

    print("\n======================================")
    print(f"Processing Aspect {aspect} ({name})")
    print("======================================")

    save_path = os.path.join(
        BASE_DATA,
        f"processed_aspect_{aspect}"
    )

    os.makedirs(save_path, exist_ok=True)

    # =========================================================
    # TRAIN DATA
    # reviews.aspectX.train.txt.gz -> train.jsonl
    # =========================================================

    train_file = os.path.join(
        BASE_DATA,
        f"reviews.aspect{aspect}.train.txt.gz"
    )

    train_examples = []

    print(f"\nLoading training data:")
    print(train_file)

    with gzip.open(train_file, "rt", encoding="utf-8") as f:

        for line in f:

            labels, text = line.split('\t')

            labels = [float(v) for v in labels.split()]

            # Convert score -> binary label
            if labels[aspect] <= neg_thres:
                label = 0

            elif labels[aspect] >= pos_thres:
                label = 1

            else:
                continue

            tokens = text.strip().split()

            train_examples.append([
                tokens,
                label
            ])

    print(f"Training examples: {len(train_examples)}")

    train_jsonl = os.path.join(save_path, "train.jsonl")

    with open(train_jsonl, "w", encoding="utf-8") as f:

        for ex in train_examples:

            f.write(json.dumps(ex) + "\n")

    print(f"Saved: {train_jsonl}")

    # =========================================================
    # VALIDATION DATA
    # reviews.aspectX.heldout.txt.gz -> valid.jsonl
    # =========================================================

    valid_file = os.path.join(
        BASE_DATA,
        f"reviews.aspect{aspect}.heldout.txt.gz"
    )

    valid_examples = []

    print(f"\nLoading validation data:")
    print(valid_file)

    with gzip.open(valid_file, "rt", encoding="utf-8") as f:

        for line in f:

            labels, text = line.split('\t')

            labels = [float(v) for v in labels.split()]

            # Convert score -> binary label
            if labels[aspect] <= neg_thres:
                label = 0

            elif labels[aspect] >= pos_thres:
                label = 1

            else:
                continue

            tokens = text.strip().split()

            valid_examples.append([
                tokens,
                label
            ])

    print(f"Validation examples: {len(valid_examples)}")

    valid_jsonl = os.path.join(save_path, "valid.jsonl")

    with open(valid_jsonl, "w", encoding="utf-8") as f:

        for ex in valid_examples:

            f.write(json.dumps(ex) + "\n")

    print(f"Saved: {valid_jsonl}")

    # =========================================================
    # TEST DATA
    # annotations.json -> test.jsonl
    # =========================================================

    print("\nLoading annotation test data...")

    test_examples = []

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)

            y = item["y"][aspect]

            # Convert score -> binary label
            if y <= neg_thres:
                label = 0

            elif y >= pos_thres:
                label = 1

            else:
                continue

            rationale = item[str(aspect)]

            # Skip samples without rationale
            if len(rationale) == 0:
                continue

            tokens = item["x"]

            test_examples.append([
                tokens,
                label,
                rationale
            ])

    print(f"Test examples: {len(test_examples)}")

    test_jsonl = os.path.join(save_path, "test.jsonl")

    with open(test_jsonl, "w", encoding="utf-8") as f:

        for ex in test_examples:

            f.write(json.dumps(ex) + "\n")

    print(f"Saved: {test_jsonl}")

print("\n======================================")
print("Completed preprocessing all aspects")
print("======================================")