import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Separating pure text...")

with open(f"./data/datasets/{args.dataset}/punctuation_cleaned.json", "r") as f:
    data = json.load(f)

text = ""
for sample in data:
    text += sample["script"]

with open(f"./data/datasets/{args.dataset}/pure_text.txt", "w") as f:
    f.write(text)
