import json

import numpy as np

PATH = "./data/sentence_transformer_model/"
# load train dataset
with open(f"{PATH}train_dataset.json", "r") as f:
    train_dataset = json.load(f)

# save the one-hot encoded genres for the train dataset as a numpy array
np.save(f"{PATH}y_train_labels.npy", np.array([ sample["genre_one-hot"] for sample in train_dataset ]))

# load test dataset
with open(f"{PATH}test_dataset.json", "r") as f:
    test_dataset = json.load(f)

# save the one-hot encoded genres for the test dataset as a numpy array
np.save(f"{PATH}y_test_labels.npy", np.array([ sample["genre_one-hot"] for sample in test_dataset ]))
