import argparse
import os
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Cleaning characters...")

with open(f"./data/scraped_data/scraped_{args.dataset}_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    # removal of some character combinations that are not useful
    sample["script"] = re.sub(r"\\'", "'", sample["script"])
    sample["script"] = re.sub(r"((\")|(\\t)|(\[')|('\])|(--)|(\[)|(\]))", "", sample["script"])
    sample["script"] = re.sub(r"((\s+)|(\\r\\n)+)", " ", sample["script"])
    sample["script"] = re.sub(r"((,\s'\s',\s)|(\s',\s)|(\s'\s)|(\s,\s))", " ", sample["script"])
    sample["script"] = re.sub(r"((\s')|('\s))", " ", sample["script"])
    # converting whole words in uppercase to lowercase
    sample["script"] = re.sub(r"[A-Z]{2,}", lambda x: x.group().lower(), sample["script"])

os.makedirs(f"./data/datasets/{args.dataset}", exist_ok=True)
with open(f"./data/datasets/{args.dataset}/characters_cleaned.json", "w") as f:
    json.dump(data, f, indent=2)
