import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Cleaning descriptions...")

with open(f"./data/datasets/{args.dataset}/names_cleaned.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = sample["script"][150:]
    
    sample["script"] += ' '
    while sample["script"][0] != " ":
        sample["script"] = sample["script"][1:]

    sample["script"] = sample["script"][1:-1]

with open(f"./data/datasets/{args.dataset}/description_cleaned.json", "w") as f:
    json.dump(data, f, indent=2)
