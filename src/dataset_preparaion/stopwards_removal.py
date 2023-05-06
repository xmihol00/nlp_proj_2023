import nltk
import json

stopwords = set(nltk.corpus.stopwords.words("english"))
with open("./data/datasets/description_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = " ".join([word for word in sample["script"].split() if word not in stopwords])

with open("./data/datasets/stopwords_removed_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)

