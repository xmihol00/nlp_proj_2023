import json

with open("./data/datasets/stopwords_removed_imsdb_data.json", "r") as f:
    data = json.load(f)

number_of_samples = len(data)
train_dataset_size = int(number_of_samples * 0.9)
train_dataset = data[:train_dataset_size]
test_dataset = data[train_dataset_size:]

with open("./data/statistical_model/train_dataset.json", "w") as f:
    json.dump(train_dataset, f, indent=2)

with open("./data/statistical_model/test_dataset.json", "w") as f:
    json.dump(test_dataset, f, indent=2)
