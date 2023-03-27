import re
import json

with open("./data/html_tags_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"\s+", " ", re.sub(r"(\s*\.\s+)|(\s*,\s+)|(\s+'\s+)", " ", sample["script"]))

with open("./data/punctuation_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
