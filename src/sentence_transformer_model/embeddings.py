import argparse
import numpy as np
import json
from sentence_transformers import SentenceTransformer

path = "./data/sentence_transformer_model/"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="all-mpnet-base-v2", help="SentenceTransformer model to use")
args = parser.parse_args()

model = SentenceTransformer(args.model, device="cpu")
print(f"Using model: {args.model}")

# load train dataset
with open(f"{path}train_dataset.json", "r") as f:
    train_dataset = json.load(f)

# get the shape of the converted train dataset and create a numpy array with the same shape
train_dataset_embeddings = np.zeros((len(train_dataset), model.get_sentence_embedding_dimension()))

# convert each sample in the train dataset to an embedding
for i, sample in enumerate(train_dataset):
    train_dataset_embeddings[i] = model.encode(sample["script"])
    if i % 100 == 0:
        print(f"Converted {i} samples to embeddings")

# save the train dataset embeddings
np.save(f"{path}X_train_embeddings_{args.model}.npy", train_dataset_embeddings)

# load test dataset
with open(f"{path}test_dataset.json", "r") as f:
    test_dataset = json.load(f)

# get the shape of the converted test dataset and create a numpy array with the same shape
test_dataset_embeddings = np.zeros((len(test_dataset), model.get_sentence_embedding_dimension()))

# convert each sample in the test dataset to an embedding
for i, sample in enumerate(test_dataset):
    test_dataset_embeddings[i] = model.encode(sample["script"])

# save the test dataset embeddings
np.save(f"{path}X_test_embeddings_{args.model}.npy", test_dataset_embeddings)

# convert the test dataset of whole scripts to embeddings
with open(f"{path}test_dataset_whole_scripts.json", "r") as f:
    test_dataset_whole_scripts = json.load(f)

test_dataset_whole_scripts_embeddings = []
for sample in test_dataset_whole_scripts:
    converted_sample = { "title": sample["title"], "genre": sample["genre"], "genre_one-hot": sample["genre_one-hot"], "embeddings": [] }
    for script in sample["script"]:
        converted_sample["embeddings"].append(model.encode(script).tolist())
    test_dataset_whole_scripts_embeddings.append(converted_sample)

# save the test dataset of whole scripts as a json file
with open(f"{path}test_whole_scripts_{args.model}.json", "w") as f:
    json.dump(test_dataset_whole_scripts_embeddings, f, indent=2)
    
