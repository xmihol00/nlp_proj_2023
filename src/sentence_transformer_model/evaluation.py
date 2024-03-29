import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import hash_model_attributes

def evaluate(model_name: str, genres: list[str], model_dataset: str = "imsdb", evaluation_dataset: str = "imsdb"):
    # load genres
    with open(f"./data/datasets/genres.json", "r") as f:
        all_genres = json.load(f)

    if len(genres) == 0:
        genres = all_genres
    else:
        genres = list(set(map(lambda x: x.strip(), genres)))
        genres = [ genre for genre in genres if genre in all_genres ]

    path = f"./models/{hash_model_attributes(model_name, genres, model_dataset)}"

    # load model and config
    model = tf.keras.models.load_model(f"{path}/model.h5")
    with open(f"{path}/config.json", "r") as f:
        config = json.load(f)
    genres = config["genres"]

    # load test dataset with whole scripts
    with open(f"./data/sentence_transformer_model/{evaluation_dataset}/test_whole_scripts_{model_name}.json", "r") as f:
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
        predicted_truth["truth_genres"] = [ genre for genre in script["genres"] if genre in genres ]

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

    average_metrics = {}
    average_metrics["Average IoU"] = total_IoU / total_predicted
    average_metrics["Average recall"] = total_recall / total_predicted
    average_metrics["Average precision"] = total_precision / total_predicted
    average_metrics["Average F1"] = total_F1 / total_predicted
    average_metrics["samples"] = total_predicted

    # load current metrics
    metrics = {}
    if os.path.exists(f"{path}/metrics.json"):
        with open(f"{path}/metrics.json", "r") as f:
            metrics = json.load(f)

    metrics[evaluation_dataset] = average_metrics

    # save average metrics
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # save the results
    with open(f"{path}/predicted_truth.json", "w") as f:
        json.dump(predicted_truth_set, f, indent=2)
    
    return total_predicted, average_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="all-mpnet-base-v2", 
                        choices=["all-mpnet-base-v2", "all-MiniLM-L12-v2", "all-distilroberta-v1", "multi-qa-mpnet-base-dot-v1", "average_word_embeddings_glove.6B.300d"],
                        help="SentenceTransformer model to use.")
    parser.add_argument("-g", "--genres", nargs='+', type=str, default=[],
                        help="Genres to evaluate on separated by comma, unknown genres will be removed. If empty, evaluate on all available genres.")
    parser.add_argument("-md", "--model_dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset, on which the model was trained on.")
    parser.add_argument("-ed", "--evaluation_dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset to evaluate on.")
    args = parser.parse_args()

    total_predicted, average_metrics = evaluate(args.model, args.genres, args.model_dataset, args.evaluation_dataset)
    print(f"Predicted samples: {total_predicted}")
    print(f"Average IoU: {average_metrics['Average IoU']}")
    print(f"Average recall: {average_metrics['Average recall']}")
    print(f"Average precision: {average_metrics['Average precision']}")
    print(f"Average F1: {average_metrics['Average F1']}")    
