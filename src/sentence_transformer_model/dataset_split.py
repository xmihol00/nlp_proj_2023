import argparse
import json
import os
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from seed import RANDOM_SEED

def split_dataset(dataset: str):
    os.makedirs(f"./data/sentence_transformer_model/{dataset}", exist_ok=True)

    with open(f"./data/datasets/{dataset}/final.json", "r") as f:
        data = json.load(f)

    with open(f"./data/datasets/genres.json", "r") as f:
        genres = set(json.load(f))

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
            train_dataset_splitted.append({ "title": sample["title"], "script": " ".join(script[i:i+256]), "genres": sorted(sample["genres"]) })
            genres.update(sample["genres"])

    genres = sorted(list(genres))

    # shuffle the data
    random.seed(RANDOM_SEED)
    random.shuffle(train_dataset_splitted)

    # convert the genres to a one-hot encoding
    for sample in train_dataset_splitted:
        sample["genre_one-hot"] = [ 1 if genre in sample["genres"] else 0 for genre in genres ]

    with open(f"./data/sentence_transformer_model/{dataset}/train_dataset.json", "w") as f:
        json.dump(train_dataset_splitted, f, indent=2)

    # do the same for test set
    test_dataset_splitted = []
    for sample in test_dataset:
        script = sample["script"].split()
        for i in range(0, len(script), 256):
            test_dataset_splitted.append({ "title": sample["title"], "script": " ".join(script[i:i+256]), "genres": sorted(sample["genres"]) })

    for sample in test_dataset_splitted:
        sample["genre_one-hot"] = [ 1 if genre in sample["genres"] else 0 for genre in genres ]

    with open(f"./data/sentence_transformer_model/{dataset}/test_dataset.json", "w") as f:
        json.dump(test_dataset_splitted, f, indent=2)

    # create the test set also with multiple 256 word samples per script
    test_dataset_splitted = []
    for sample in test_dataset:
        script = sample["script"].split()
        new_samples = { "title": sample["title"], "script": [], "genres": sorted(sample["genres"])}
        for i in range(0, len(script), 256):
            new_samples["script"].append(" ".join(script[i:i+256]))
        test_dataset_splitted.append(new_samples)
        new_samples["genre_one-hot"] = [ 1 if genre in sample["genres"] else 0 for genre in genres ]

    with open(f"./data/sentence_transformer_model/{dataset}/test_dataset_whole_scripts.json", "w") as f:
        json.dump(test_dataset_splitted, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset to split.")
    args = parser.parse_args()
    
    split_dataset(args.dataset)
