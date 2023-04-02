from nltk.stem.snowball import SnowballStemmer
import json

with open("./data/datasets/stopwords_removed_imsdb_data.json", "r") as f:
    data = json.load(f)

stemmer = SnowballStemmer("english")
for sample in data:
    sample["script"] = " ".join([stemmer.stem(word) for word in sample["script"].split()])

with open("./data/datasets/stemmed_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)