import argparse
import json
import numpy as np

def encode_labels(dataset: str):
    # load train dataset
    with open(f"./data/sentence_transformer_model/{dataset}/train_dataset.json", "r") as f:
        train_dataset = json.load(f)

    # save the one-hot encoded genres for the train dataset as a numpy array
    np.save(f"./data/sentence_transformer_model/{dataset}/y_train_labels.npy", np.array([ sample["genre_one-hot"] for sample in train_dataset ]))

    # load test dataset
    with open(f"./data/sentence_transformer_model/{dataset}/test_dataset.json", "r") as f:
        test_dataset = json.load(f)

    # save the one-hot encoded genres for the test dataset as a numpy array
    np.save(f"./data/sentence_transformer_model/{dataset}/y_test_labels.npy", np.array([ sample["genre_one-hot"] for sample in test_dataset ]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                        help="Dataset to evaluate on.")
    args = parser.parse_args()
    encode_labels(args.dataset)