import re
import json

with open("./data/datasets/scraped_imsdb_data.json", "r") as f:
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

with open("./data/datasets/characters_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
