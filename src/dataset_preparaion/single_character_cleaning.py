import re
import json

with open("./data/datasets/stopwords_removed_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"\s.\s", " ", sample["script"])

with open("./data/datasets/single_character_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)

