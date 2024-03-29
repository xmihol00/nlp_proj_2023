import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import hash_model_attributes

def train(model_name: str, genres: list[str], dataset: str):
    # load genres
    with open(f"./data/datasets/genres.json", "r") as f:
        all_genres = json.load(f)

    config = {}
    if len(genres) == 0:
        genres_indices = list(range(len(all_genres)))
        dir_name = hash_model_attributes(model_name, all_genres, dataset)
    else:
        picked_genres = sorted(list(set(map(lambda x: x.strip(), genres))))
        picked_genres = [ genre for genre in picked_genres if genre in all_genres ]
        genres_indices = [ all_genres.index(genre) for genre in picked_genres ]
        dir_name = hash_model_attributes(model_name, picked_genres, dataset)
        all_genres = picked_genres

    config["genres"] = all_genres
    config["genres_indices"] = genres_indices
    config["model"] = model_name
    config["hash"] = dir_name
    config["dataset"] = dataset
    
    # load train dataset
    X_train = np.load(f"./data/sentence_transformer_model/{dataset}/X_train_embeddings_{model_name}.npy")
    y_train = np.load(f"./data/sentence_transformer_model/{dataset}/y_train_labels.npy")

    # load test dataset
    X_test = np.load(f"./data/sentence_transformer_model/{dataset}/X_test_embeddings_{model_name}.npy")
    y_test = np.load(f"./data/sentence_transformer_model/{dataset}/y_test_labels.npy")

    # select only the genres to train on
    y_train = y_train[:, genres_indices]
    y_test = y_test[:, genres_indices]

    # print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")


    # model
    HIDDEN_LAYERS = 4
    if HIDDEN_LAYERS == 2:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid")
            ]
        )
    elif HIDDEN_LAYERS == 3:
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
    elif HIDDEN_LAYERS == 4:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(X_train.shape[1], activation="relu", input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.25),
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

    # save the model and config
    full_dir_name = f"./models/{dir_name}"
    os.makedirs(full_dir_name, exist_ok=True)
    model.save(f"{full_dir_name}/model.h5")
    with open(f"{full_dir_name}/config.json", "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="all-mpnet-base-v2", 
                        choices=["all-mpnet-base-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1", "multi-qa-mpnet-base-dot-v1", "average_word_embeddings_glove.6B.300d"],
                        help="Sentence transformer model to use.")
    parser.add_argument("-g", "--genres", nargs='+', type=str, default=[],
                        help="Genres to train on separated by comma, unknown genres will be removed. If empty, train on all available genres.")
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset to train on.")
    args = parser.parse_args()

    train(args.model, args.genres, args.dataset)
