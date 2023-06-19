import argparse
from nltk.stem.snowball import SnowballStemmer
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Stemming...")

with open(f"./data/datasets/{args.dataset}/stopwords_removed.json", "r") as f:
    data = json.load(f)

stemmer = SnowballStemmer("english")
for sample in data:
    sample["script"] = " ".join([stemmer.stem(word) for word in sample["script"].split()])

with open(f"./data/datasets/{args.dataset}/stemmed.json", "w") as f:
    json.dump(data, f, indent=2)