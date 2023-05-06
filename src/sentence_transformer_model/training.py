import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import hash_model_name_and_labels

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="all-mpnet-base-v2", 
                    choices=["all-mpnet-base-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1", "multi-qa-mpnet-base-dot-v1", "average_word_embeddings_glove.6B.300d"],
                    help="Sentence transformer model to use.")
parser.add_argument("-g", "--genres", type=str, default="", 
                    help="Genres to train on separated by comma, unknown genres will be removed. If empty, train on all available genres.")
args = parser.parse_args()

# load genres and select indices of the genres to train on
with open("./data/sentence_transformer_model/genres.json", "r") as f:
    genres = json.load(f)

config = {}
if args.genres == "":
    genres_indices = list(range(len(genres)))
    dir_name = hash_model_name_and_labels(args.model, genres)
else:
    picked_genres = sorted(list(set(map(lambda x: x.strip(), args.genres.split(",")))))
    picked_genres = [ genre for genre in picked_genres if genre in genres ]
    genres_indices = [ genres.index(genre) for genre in picked_genres ]
    dir_name = hash_model_name_and_labels(args.model, picked_genres)
    genres = picked_genres

config["genres"] = genres
config["genres_indices"] = genres_indices
config["model"] = args.model
config["hash"] = dir_name

full_dir_name = f"models/sentence_transformer/{dir_name}/"
os.makedirs(full_dir_name, exist_ok=True)

# load train dataset
X_train = np.load(f"./data/sentence_transformer_model/X_train_embeddings_{args.model}.npy")
y_train = np.load(f"./data/sentence_transformer_model/y_train_labels.npy")

# load test dataset
X_test = np.load(f"./data/sentence_transformer_model/X_test_embeddings_{args.model}.npy")
y_test = np.load(f"./data/sentence_transformer_model/y_test_labels.npy")

# select only the genres to train on
y_train = y_train[:, genres_indices]
y_test = y_test[:, genres_indices]
# TODO: remove samples with all zeros

# print shapes
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid")
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# get the weights of the model before training
initial_weights = model.get_weights()
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.15, 
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                                patience=5, restore_best_weights=True, monitor="val_accuracy", mode="max")]).history

# get the best epoch
best_epoch = np.argmax(history["val_accuracy"]) + 1

# train on the full train dataset with the best epoch
model.set_weights(initial_weights)
model.fit(X_train, y_train, epochs=best_epoch, batch_size=128)

# evaluate on the test dataset
loss, accuracy = model.evaluate(X_test, y_test)

# save the model and config
model.save(f"{full_dir_name}model.h5")
with open(f"{full_dir_name}config.json", "w") as f:
    json.dump(config, f, indent=2)
