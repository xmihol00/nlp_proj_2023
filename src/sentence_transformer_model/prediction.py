import argparse
import os
import re
import sys
import numpy as np
import tensorflow as tf
import json
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import hash_model_attributes, preprocess_for_embedding
    
def predict_string(model_name: str, genres: list[str], dataset: str, script: str, number_of_classes: int = 2):
    # load genres
    with open(f"./data/datasets/genres.json", "r") as f:
        all_genres = json.load(f)

    if len(genres) == 0:
        genres = all_genres
    else:
        genres = sorted(list(set(map(lambda x: x.strip(), genres))))
        genres = [ genre for genre in genres if genre in all_genres ]

    path = f"./models/{hash_model_attributes(model_name, genres, dataset)}"

    # load config
    with open(f"{path}/config.json", "r") as f:
        config = json.load(f)
    genres = config["genres"]

    # load model
    model = tf.keras.models.load_model(f"{path}/model.h5")

    # load sentence transformer model
    sentence_transformer = SentenceTransformer(model_name, device="cpu")

    script = preprocess_for_embedding(script)

    # split to parts of 256 tokens
    script_parts = []
    script = script.split()
    for i in range(0, len(script), 256):
        script_parts.append(' '.join(script[i:i+256]))

    # create embeddings
    script_embeddings = sentence_transformer.encode(script_parts)
    if script_embeddings.shape[0] == 0:
        return []

    # predict
    predictions = model.predict(script_embeddings)

    # sum probabilities and normalized
    predictions = np.sum(predictions, axis=0) / predictions.shape[0]
    predictions_argsorted = np.argsort(predictions)[::-1][:number_of_classes]
    genres = [ genres[idx] for idx in predictions_argsorted ]

    return genres


def predict_file(model_name: str, genres: list[str], dataset: str, script_file_name: str, number_of_classes: int = 2):
    # load script
    with open(script_file_name, "r") as f:
        script = f.read()
    
    return predict_string(model_name, genres, dataset, script, number_of_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="Name of the file with a movies script to be classified.")
    parser.add_argument("-m", "--model", type=str, default="all-mpnet-base-v2", 
                        choices=["all-mpnet-base-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1", "multi-qa-mpnet-base-dot-v1", "average_word_embeddings_glove.6B.300d"],
                        help="Sentence transformer model to use.")
    parser.add_argument("-g", "--genres", nargs='+', type=str, default=[],
                        help="Genres to train on separated by comma, unknown genres will be removed. If empty, train on all available genres.")
    parser.add_argument("-n", "--number_of_classes", default=2, type=int, help="Number of classes to predict.")
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset on which the model was trained on.")
    args = parser.parse_args()

    genres = predict_file(args.model, args.genres, args.dataset, args.file_name, args.number_of_classes)
    print(f"Predicted genres: {genres}")
