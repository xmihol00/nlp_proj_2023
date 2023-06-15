import json
import os

def merge_datasets(data_imsdb, data_dailyscript):
    for dailyscript_sample in data_dailyscript:
        # check if sample is already in imsdb
        already_in_imsdb = False
        for i, imsdb_sample in enumerate(data_imsdb):
            if imsdb_sample["title"] == dailyscript_sample["title"]:
                already_in_imsdb = True
                break
        
        if not already_in_imsdb:
            data_imsdb.append(dailyscript_sample)
        else:
            data_imsdb[i]["genres"] = list(set(dailyscript_sample["genres"] + data_imsdb[i]["genres"]))
    
    return data_imsdb

os.makedirs("./data/datasets/merged", exist_ok=True)
print("Merging datasets...")

with open("./data/datasets/imsdb/final_stemmed_no_stopwords.json", "r") as f:
    data_imsdb = json.load(f)
with open("./data/datasets/dailyscript/final_stemmed_no_stopwords.json", "r") as f:
    data_dailyscript = json.load(f)

merged = merge_datasets(data_imsdb, data_dailyscript)
with open("./data/datasets/merged/final_stemmed_no_stopwords.json", "w") as f:
    json.dump(merged, f, indent=2)

with open("./data/datasets/imsdb/final.json", "r") as f:
    data_imsdb = json.load(f)
with open("./data/datasets/dailyscript/final.json", "r") as f:
    data_dailyscript = json.load(f)

merged = merge_datasets(data_imsdb, data_dailyscript)
with open("./data/datasets/merged/final.json", "w") as f:
    json.dump(merged, f, indent=2)
