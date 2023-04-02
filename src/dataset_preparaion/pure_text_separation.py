import re
import json

with open("./data/datasets/punctuation_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

text = ""
for sample in data:
    text += sample["script"]

with open("./data/datasets/pure_text_imsdb_data.txt", "w") as f:
    f.write(text)
