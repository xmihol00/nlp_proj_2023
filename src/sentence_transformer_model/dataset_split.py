import json
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from seed import RANDOM_SEED

def imsdb_dataset_split():
    os.makedirs("./data/sentence_transformer_model", exist_ok=True)

    with open("./data/datasets/final_imsdb_data.json", "r") as f:
        data = json.load(f)

    # get all genres to use as labels
    genres = set()
    for sample in data:
        genres.update(sample["genre"])

    with open("./data/sentence_transformer_model/genres.json", "w") as f:
        json.dump(sorted(list(genres)), f, indent=2)

    # split the data to train and test datasets 90/10
    number_of_samples = len(data)
    train_dataset_size = int(number_of_samples * 0.9)
    train_dataset = data[:train_dataset_size]
    test_dataset = data[train_dataset_size:] 

    # split each sample to multiple samples each with 256 words and collect all genres (labels)
    train_dataset_splitted = []
    for sample in train_dataset:
        script = sample["script"].split()
        for i in range(0, len(script), 256):
            train_dataset_splitted.append({ "title": sample["title"], "script": " ".join(script[i:i+256]), "genre": sorted(sample["genre"]) })
            genres.update(sample["genre"])

    genres = sorted(list(genres))

    # shuffle the data
    random.seed(RANDOM_SEED)
    random.shuffle(train_dataset_splitted)

    # convert the genres to a one-hot encoding
    for sample in train_dataset_splitted:
        sample["genre_one-hot"] = [ 1 if genre in sample["genre"] else 0 for genre in genres ]

    with open("./data/sentence_transformer_model/train_dataset.json", "w") as f:
        json.dump(train_dataset_splitted, f, indent=2)

    # do the same for test set
    test_dataset_splitted = []
    for sample in test_dataset:
        script = sample["script"].split()
        for i in range(0, len(script), 256):
            test_dataset_splitted.append({ "title": sample["title"], "script": " ".join(script[i:i+256]), "genre": sorted(sample["genre"]) })

    for sample in test_dataset_splitted:
        sample["genre_one-hot"] = [ 1 if genre in sample["genre"] else 0 for genre in genres ]

    with open("./data/sentence_transformer_model/test_dataset.json", "w") as f:
        json.dump(test_dataset_splitted, f, indent=2)

    # create the test set also with multiple 256 word samples per script
    test_dataset_splitted = []
    for sample in test_dataset:
        script = sample["script"].split()
        new_samples = { "title": sample["title"], "script": [], "genre": sorted(sample["genre"])}
        for i in range(0, len(script), 256):
            new_samples["script"].append(" ".join(script[i:i+256]))
        test_dataset_splitted.append(new_samples)
        new_samples["genre_one-hot"] = [ 1 if genre in sample["genre"] else 0 for genre in genres ]

    with open("./data/sentence_transformer_model/test_dataset_whole_scripts.json", "w") as f:
        json.dump(test_dataset_splitted, f, indent=2)

def dailyscript_dataset_split():
    #TODO
    pass

if __name__ == "__main__":
    imsdb_dataset_split()
    dailyscript_dataset_split()
    