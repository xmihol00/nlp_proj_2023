import json

with open("./data/datasets/single_character_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

# remove trialing spaces from labels and data, remove empty labels
for sample in data:
    sample["script"] = sample["script"].strip()
    # split genres by dot and strip spaces
    sample["genre"] = list(set([ genre.strip() for genre_str in sample["genre"] for genre in genre_str.split(".") if genre.strip() != "" ]))

with open("./data/datasets/final_stemmed_no_stopwords_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)

with open("./data/datasets/description_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

# remove trialing spaces from labels and data, remove empty labels
for sample in data:
    sample["script"] = sample["script"].strip()
    # split genres by dot and strip spaces
    sample["genre"] = list(set([ genre.strip() for genre_str in sample["genre"] for genre in genre_str.split(".") if genre.strip() != "" ]))

with open("./data/datasets/final_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)

