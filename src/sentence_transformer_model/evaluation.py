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
                    help="SentenceTransformer model to use")
parser.add_argument("-g", "--genres", type=str, default="", 
                    help="Genres to evaluate on separated by comma, unknown genres will be removed. If empty, evaluate on all available genres.")
args = parser.parse_args()

# load genres
with open("./data/sentence_transformer_model/genres.json", "r") as f:
    genres_all = json.load(f)

if args.genres == "":
    genres = genres_all
else:
    genres = list(set(map(lambda x: x.strip(), args.genres.split(','))))
    genres = [ genre for genre in genres if genre in genres_all ]

full_dir_name = f"./models/sentence_transformer/{hash_model_name_and_labels(args.model, genres)}/"

# load model and config
model = tf.keras.models.load_model(f"{full_dir_name}model.h5")
with open(f"{full_dir_name}config.json", "r") as f:
    config = json.load(f)
genres_indices = config["genres_indices"]
genres = config["genres"]

# load test dataset with whole scripts
with open(f"./data/sentence_transformer_model/test_whole_scripts_{args.model}.json", "r") as f:
    test_dataset = json.load(f)

# predict all samples from one script sum the probabilities and do argmax for 5 most likely genres
total_predicted = 0
total_IoU = 0
total_recall = 0
total_precision = 0
total_F1 = 0
predicted_truth_set = []
for script in test_dataset:
    X_test = np.array(script["embeddings"])
    if X_test.shape[0] == 0:
        continue
    total_predicted += 1
    y_pred = model.predict(X_test)
    y_pred = np.sum(y_pred, axis=0) / X_test.shape[0]
    y_pred_argsorted = np.argsort(y_pred)[::-1]
    
    # remove all predictions with probability less than 0.5
    y_pred = [ idx for idx in y_pred_argsorted if y_pred[idx] > 0.5 ]
    predicted_truth = {}
    predicted_truth["title"] = script["title"]
    predicted_truth["predicted_genres"] = [ genres[idx] for idx in y_pred ]
    predicted_truth["truth_genres"] = [ genre for genre in script["genre"] if genre in genres ]

    # calculate metrics
    if len(set(predicted_truth["predicted_genres"]).union(set(predicted_truth["truth_genres"]))) == 0:
        predicted_truth["IoU"] = 1
    else:
        predicted_truth["IoU"] = (len(set(predicted_truth["predicted_genres"]).intersection(set(predicted_truth["truth_genres"]))) / 
                                  len(set(predicted_truth["predicted_genres"]).union(set(predicted_truth["truth_genres"]))))
    if len(predicted_truth["truth_genres"]) == 0:
        predicted_truth["recall"] = 1
    else:
        predicted_truth["recall"] = (len(set(predicted_truth["predicted_genres"]).intersection(set(predicted_truth["truth_genres"]))) /
                                     len(set(predicted_truth["truth_genres"])))
    if len(predicted_truth["predicted_genres"]) == 0:
        predicted_truth["precision"] = 1 if len(predicted_truth["truth_genres"]) == 0 else 0
    else:
        predicted_truth["precision"] = (len(set(predicted_truth["predicted_genres"]).intersection(set(predicted_truth["truth_genres"]))) /
                                        len(set(predicted_truth["predicted_genres"])))
    if predicted_truth["recall"] + predicted_truth["precision"] == 0:
        predicted_truth["F1"] = 0
    else:
        predicted_truth["F1"] = (2 * predicted_truth["recall"] * predicted_truth["precision"]) / (predicted_truth["recall"] + predicted_truth["precision"])

    total_IoU += predicted_truth["IoU"]
    total_recall += predicted_truth["recall"]
    total_precision += predicted_truth["precision"]
    total_F1 += predicted_truth["F1"]
    predicted_truth_set.append(predicted_truth)

# print summary
print(f"Predicted samples: {total_predicted}")
print(f"Average IoU: {total_IoU / total_predicted}")
print(f"Average recall: {total_recall / total_predicted}")
print(f"Average precision: {total_precision / total_predicted}")
print(f"Average F1: {total_F1 / total_predicted}")

average_metrics = {}
average_metrics["IoU"] = total_IoU / total_predicted
average_metrics["recall"] = total_recall / total_predicted
average_metrics["precision"] = total_precision / total_predicted
average_metrics["F1"] = total_F1 / total_predicted

# save average metrics
with open(f"{full_dir_name}metrics.json", "w") as f:
    json.dump(average_metrics, f, indent=2)

# save the results
with open(f"{full_dir_name}predicted_truth.json", "w") as f:
    json.dump(predicted_truth_set, f, indent=2)
