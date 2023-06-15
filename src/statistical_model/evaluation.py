import argparse
import json
import os

def evaluate(dataset: str):
    test_datasets = []
    all_genres_word_counts = []

    if dataset == "all":
        with open("./data/statistical_model/imsdb/test_dataset.json", "r") as f:
            test_datasets.append(json.load(f))
        with open("./models/statistical/imsdb/genres_word_counts.json", "r") as f:
            all_genres_word_counts.append(json.load(f))
        with open("./data/statistical_model/dailyscript/test_dataset.json", "r") as f:
            test_datasets.append(json.load(f))
        with open("./models/statistical/dailyscript/genres_word_counts.json", "r") as f:
            all_genres_word_counts.append(json.load(f))
    else:
        with open(f"./data/statistical_model/{dataset}/test_dataset.json", "r") as f:
            test_datasets.append(json.load(f))
        with open(f"./models/statistical/{dataset}/genres_word_counts.json", "r") as f:
            all_genres_word_counts.append(json.load(f))

    results = []
    total_IoU = 0
    total_recall = 0
    total_precision = 0
    total_F1 = 0
    total_predicted = 0

    for test_dataset, genres_word_counts in zip(test_datasets, all_genres_word_counts):
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
    os.makedirs(f"./models/statistical/{dataset}", exist_ok=True)
    with open(f"./models/statistical/{dataset}/metrics.json", "w") as f:
        json.dump(average_metrics, f, indent=2)

    with open(f"./models/statistical/{dataset}/predicted_truth.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return total_predicted, average_metrics, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "all"],
                        help="Dataset to evaluate on.")
    args = parser.parse_args()
    
    total_predicted, average_metrics, results = evaluate(args.dataset)

    # print summary
    print(f"Predicted samples: {total_predicted}")
    print(f"Average IoU: {average_metrics['IoU']}")
    print(f"Average recall: {average_metrics['recall']}")
    print(f"Average precision: {average_metrics['precision']}")
    print(f"Average F1: {average_metrics['F1']}")
