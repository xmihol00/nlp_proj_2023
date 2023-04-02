import re
import json

with open("./data/datasets/special_cases_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"[A-Z]\w*", "", sample["script"])
    sample["script"] = re.sub(r"'s", "", sample["script"])
    sample["script"] = re.sub(r"\s+", " ", sample["script"])

with open("./data/datasets/names_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
