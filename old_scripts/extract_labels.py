import os
import json

json_dir = "dacl10k_dataset/annotations/train"

labels = set()
for fname in os.listdir(json_dir):
    if fname.endswith(".json"):
        with open(os.path.join(json_dir, fname), "r") as f:
            data = json.load(f)
            for shape in data.get("shapes", []):
                labels.add(shape["label"].lower())

print("Unique labels in dataset:", labels)