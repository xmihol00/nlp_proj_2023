import argparse
import json
import numpy as np

def encode_labels(path: str):
    # load train dataset
    with open(f"{path}train_dataset.json", "r") as f:
        train_dataset = json.load(f)

    # save the one-hot encoded genres for the train dataset as a numpy array
    np.save(f"{path}y_train_labels.npy", np.array([ sample["genre_one-hot"] for sample in train_dataset ]))

    # load test dataset
    with open(f"{path}test_dataset.json", "r") as f:
        test_dataset = json.load(f)

    # save the one-hot encoded genres for the test dataset as a numpy array
    np.save(f"{path}y_test_labels.npy", np.array([ sample["genre_one-hot"] for sample in test_dataset ]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="./data/sentence_transformer_model/", help="path to data (with trailing slash)")
    args = parser.parse_args()
    encode_labels(args.path)