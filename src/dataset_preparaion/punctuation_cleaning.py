import re
import json

with open("./data/datasets/html_tags_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"(\.|,|\?|\!|\))\s+[A-Z]", lambda x: x.group().lower(), sample["script"]) 
    sample["script"] = re.sub(r"((\.+\s+)|(\s+\.+)|(\s*,\s+)|(\s+')|('\s+)|(:\s+)|(;\s+)|" + 
                              r"(\?)|(\!))", " ", sample["script"])
    sample["script"] = re.sub(r"((\.\.\.)|(\.\.)|(-\s)|(\s-))", " ", sample["script"])
    sample["script"] = re.sub(r"(\.|,|:|;|-)", " ", sample["script"])
    sample["script"] = re.sub(r"(\s+)", " ", sample["script"])

with open("./data/datasets/punctuation_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
