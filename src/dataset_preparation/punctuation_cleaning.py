import argparse
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Cleaning punctuation...")

with open(f"./data/datasets/{args.dataset}/html_tags_cleaned.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"(\.|,|\?|\!|\))\s+[A-Z]", lambda x: x.group().lower(), sample["script"]) 
    sample["script"] = re.sub(r"((\.+\s+)|(\s+\.+)|(\s*,\s+)|(\s+')|('\s+)|(:\s+)|(;\s+)|" + 
                              r"(\?)|(\!))", " ", sample["script"])
    sample["script"] = re.sub(r"((\.\.\.)|(\.\.)|(-\s)|(\s-))", " ", sample["script"])
    sample["script"] = re.sub(r"(\.|,|:|;|-)", " ", sample["script"])
    sample["script"] = re.sub(r"(\s+)", " ", sample["script"])

with open(f"./data/datasets/{args.dataset}/punctuation_cleaned.json", "w") as f:
    json.dump(data, f, indent=2)
