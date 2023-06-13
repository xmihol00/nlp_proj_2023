import json
import os

def imsdb_dataset_split():
    with open("./data/datasets/cleaned_preprocessed_imsdb_data.json", "r") as f:
        data = json.load(f)

    number_of_samples = len(data)
    train_dataset_size = int(number_of_samples * 0.9)
    train_dataset = data[:train_dataset_size]
    test_dataset = data[train_dataset_size:]

    os.makedirs("./data/statistical_model", exist_ok=True)

    with open("./data/statistical_model/train_dataset.json", "w") as f:
        json.dump(train_dataset, f, indent=2)

    with open("./data/statistical_model/test_dataset.json", "w") as f:
        json.dump(test_dataset, f, indent=2)

def dailyscript_dataset_split():
    #TODO
    pass

if __name__ == "__main__":
    imsdb_dataset_split()
    dailyscript_dataset_split()
    