import argparse
import json
import os

def dataset_split(dataset: str):
    with open(f"./data/datasets/{dataset}/final_stemmed_no_stopwords.json", "r") as f:
        data = json.load(f)

    number_of_samples = len(data)
    train_dataset_size = int(number_of_samples * 0.9)
    train_dataset = data[:train_dataset_size]
    test_dataset = data[train_dataset_size:]

    genres = set()
    for sample in data:
        genres.update(sample["genres"])

    os.makedirs(f"./data/statistical_model/{dataset}", exist_ok=True)

    with open(f"./data/datasets/genres.json", "r") as f:
        genres = set(json.load(f))

    with open(f"./data/statistical_model/{dataset}/train_dataset.json", "w") as f:
        json.dump(train_dataset, f, indent=2)

    with open(f"./data/statistical_model/{dataset}/test_dataset.json", "w") as f:
        json.dump(test_dataset, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset to evaluate on.")
    args = parser.parse_args()
    
    dataset_split(args.dataset)
