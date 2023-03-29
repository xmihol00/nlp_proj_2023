import json

with open("./data/special_cases_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = sample["script"][150:]
    
    sample["script"] += ' '
    while sample["script"][0] != " ":
        sample["script"] = sample["script"][1:]

    sample["script"] = sample["script"][1:-1]

with open("./data/description_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
