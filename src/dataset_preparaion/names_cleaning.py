import argparse
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Cleaning names...")

with open(f"./data/datasets/{args.dataset}/special_cases_cleaned.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"[A-Z]\w*", "", sample["script"])
    sample["script"] = re.sub(r"'s", "", sample["script"])
    sample["script"] = re.sub(r"\s+", " ", sample["script"])

with open(f"./data/datasets/{args.dataset}/names_cleaned.json", "w") as f:
    json.dump(data, f, indent=2)
