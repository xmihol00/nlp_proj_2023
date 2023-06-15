import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Cleaning labels...")

with open(f"./data/datasets/{args.dataset}/single_character_cleaned.json", "r") as f:
    data = json.load(f)

# remove trialing spaces from labels and data, remove empty labels
for sample in data:
    sample["script"] = sample["script"].strip()
    # split genres by dot and strip spaces
    sample["genres"] = list(set([ genre.strip() for genre_str in sample["genres"] for genre in genre_str.split(".") if genre.strip() != "" ]))

with open(f"./data/datasets/{args.dataset}/final_stemmed_no_stopwords.json", "w") as f:
    json.dump(data, f, indent=2)

with open(f"./data/datasets/{args.dataset}/description_cleaned.json", "r") as f:
    data = json.load(f)

# remove trialing spaces from labels and data, remove empty labels
for sample in data:
    sample["script"] = sample["script"].strip()
    # split genres by dot and strip spaces
    sample["genres"] = list(set([ genre.strip() for genre_str in sample["genres"] for genre in genre_str.split(".") if genre.strip() != "" ]))

with open(f"./data/datasets/{args.dataset}/final.json", "w") as f:
    json.dump(data, f, indent=2)

