import re
import json

with open("./data/scraped_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"(,\s'\s',\s)|(\s',\s)|(\s'\s)|(\s,\s)|\s+", " ", 
                              re.sub(r"\s+|(\\r\\n)+", " ", 
                                     re.sub(r"(\")|(\\t)|(\[')|('\])|(--)", "", sample["script"].replace("\\'", "'"))))

with open("./data/characters_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
