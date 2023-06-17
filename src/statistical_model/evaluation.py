import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import hash_model_attributes

def evaluate(genres: list[str], model_dataset: str = "imsdb", evaluation_dataset: str = "imsdb"):
    # load genres
    with open(f"./data/datasets/genres.json", "r") as f:
        all_genres = json.load(f)

    if len(genres) == 0:
        genres = all_genres
    else:
        genres = list(set(map(lambda x: x.strip(), genres)))
        genres = [ genre for genre in genres if genre in all_genres ]

    path = f"./models/{hash_model_attributes('statistical', genres, model_dataset)}"

    with open(f"./data/statistical_model/{evaluation_dataset}/test_dataset.json", "r") as f:
        test_dataset = json.load(f)
    
    with open(f"{path}/genres_word_counts.json", "r") as f:
        genres_word_counts = json.load(f)

    results = []
    total_IoU = 0
    total_recall = 0
    total_precision = 0
    total_F1 = 0
    total_predicted = 0

    for sample in test_dataset:
        if sample["script"] == "":
            continue
        counts = { genre: 0 for genre in genres_word_counts }
        for genre in genres_word_counts:
            for word in sample["script"].split():
                if word in genres_word_counts[genre]:
                    counts[genre] += genres_word_counts[genre][word]

        ground_truth = set(genre for dirty_genre in sample["genres"] if dirty_genre.strip() != "" for genre in dirty_genre.strip().split("."))
        counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        counts = counts[:len(ground_truth)]

        prediction = set(count[0] for count in counts)
        matches = prediction & ground_truth

        predicted_truth = {}
        predicted_truth["title"] = sample["title"]
        predicted_truth["predicted_genres"] = list(prediction)
        predicted_truth["truth_genres"] = list(ground_truth)
        predicted_truth["IoU"] = len(matches) / len(prediction | ground_truth)
        predicted_truth["recall"] = len(matches) / len(ground_truth)
        predicted_truth["precision"] = len(matches) / len(prediction)
        if predicted_truth["precision"] + predicted_truth["recall"] == 0:
            predicted_truth["F1"] = 0
        else:
            predicted_truth["F1"] = 2 * predicted_truth["precision"] * predicted_truth["recall"] / (predicted_truth["precision"] + predicted_truth["recall"])

        total_IoU += predicted_truth["IoU"]
        total_recall += predicted_truth["recall"]
        total_precision += predicted_truth["precision"]
        total_F1 += predicted_truth["F1"]

        total_predicted += 1
        results.append(predicted_truth)

    average_metrics = {}
    average_metrics["IoU"] = total_IoU / total_predicted
    average_metrics["recall"] = total_recall / total_predicted
    average_metrics["precision"] = total_precision / total_predicted
    average_metrics["F1"] = total_F1 / total_predicted

    # save average metrics
    with open(f"{path}/metrics.json", "w") as f:
        json.dump(average_metrics, f, indent=2)

    with open(f"{path}/predicted_truth.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return total_predicted, average_metrics, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--genres", nargs='+', type=str, default=[],
                        help="Genres to train on separated by comma, unknown genres will be removed. If empty, train on all available genres.")
    parser.add_argument("-md", "--model_dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset, on which the model was trained on.")
    parser.add_argument("-ed", "--evaluation_dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset to evaluate on.")
    args = parser.parse_args()
    
    total_predicted, average_metrics, results = evaluate(args.genres, args.model_dataset, args.evaluation_dataset)

    # print summary
    print(f"Predicted samples: {total_predicted}")
    print(f"Average IoU: {average_metrics['IoU']}")
    print(f"Average recall: {average_metrics['recall']}")
    print(f"Average precision: {average_metrics['precision']}")
    print(f"Average F1: {average_metrics['F1']}")
