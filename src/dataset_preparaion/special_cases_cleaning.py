import re
import json

with open("./data/punctuation_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"((CLOSE ON:)|(CUT:)|(ANGLE ON:)|(\(Cont.\)\s))", "", sample["script"])
    sample["script"] = re.sub(r"(^\s+)", "", sample["script"])
    sample["script"] = re.sub(r"(\d+)", "", sample["script"])
    sample["script"] = re.sub(r"(\s+)", " ", sample["script"])
    sample["script"] = re.sub(r"(\@|\#|\$|\%|\^|\&|\*|\(|\)|\{|\}|\]|\[|\<|\>|\_|\|\~|\+|/)", "", sample["script"])
    sample["script"] = re.sub(r"\s+", " ", sample["script"])
    sample["script"] = re.sub(r" I ", " i ", sample["script"])

with open("./data/special_cases_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
