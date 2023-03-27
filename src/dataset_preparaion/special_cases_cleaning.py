import re
import json

with open("./data/punctuation_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"((CLOSE ON:)|(CUT:)|(ANGLE ON:)|(^\s+)|(\(Cont.\)\s)|(\s')|('\s)|(\.\.\.))", "", sample["script"], re.DOTALL)

with open("./data/special_cases_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
