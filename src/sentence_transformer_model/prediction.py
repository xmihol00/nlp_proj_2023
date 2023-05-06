import argparse
import os
import re
import sys
import numpy as np
import tensorflow as tf
import json
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import hash_model_name_and_labels, preprocess_for_embedding

parser = argparse.ArgumentParser()
parser.add_argument("file_name", type=str, help="Name of the file with a movies script to be classified.")
parser.add_argument("-m", "--model", type=str, default="all-mpnet-base-v2", 
                    choices=["all-mpnet-base-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1", "multi-qa-mpnet-base-dot-v1", "average_word_embeddings_glove.6B.300d"],
                    help="Sentence transformer model to use.")
parser.add_argument("-g", "--genres", type=str, default="", 
                    help="Genres to train on separated by comma, unknown genres will be removed. If empty, train on all available genres.")
parser.add_argument("-n", "--number_of_classes", default=2, type=int, help="Number of classes to predict.")
args = parser.parse_args()

with open("./data/sentence_transformer_model/genres.json", "r") as f:
    genres_all = json.load(f)

if args.genres == "":
    genres = genres_all
else:
    genres = sorted(list(set(map(lambda x: x.strip(), args.genres.split(",")))))
    genres = [ genre for genre in genres if genre in genres_all ]

# load script
with open(args.file_name, "r") as f:
    script = f.read()

path = f"./models/sentence_transformer/{hash_model_name_and_labels(args.model, genres)}/"

# load config
with open(f"{path}config.json", "r") as f:
    config = json.load(f)
genres = config["genres"]

# load model
model = tf.keras.models.load_model(f"{path}model.h5")

# load sentence transformer model
sentence_transformer = SentenceTransformer(args.model, device="cpu")

script = preprocess_for_embedding(script)

# split to parts of 256 tokens
script_parts = []
script = script.split()
for i in range(0, len(script), 256):
    script_parts.append(script[i:i+256])

# create embeddings
script_embeddings = sentence_transformer.encode(script_parts)

# predict
predictions = model.predict(script_embeddings)

# sum probabilities and normalized
predictions = np.sum(predictions, axis=0) / predictions.shape[0]
predictions_argsorted = np.argsort(predictions)[::-1][:args.number_of_classes]
genres = [ genres[idx] for idx in predictions_argsorted ]

print(f"Predicted genres: {genres}")
