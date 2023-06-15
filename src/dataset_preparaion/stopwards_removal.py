import argparse
import nltk
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Removing stopwords...")

stopwords = set(nltk.corpus.stopwords.words("english"))
with open(f"./data/datasets/{args.dataset}/description_cleaned.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = " ".join([word for word in sample["script"].split() if word not in stopwords])

with open(f"./data/datasets/{args.dataset}/stopwords_removed.json", "w") as f:
    json.dump(data, f, indent=2)

