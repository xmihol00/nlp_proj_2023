
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import preprocess_for_word_counting

def predict(file_name: str, dataset: str, number_of_classes: int = 2):    
    # load script
    with open(file_name, "r") as f:
        script = f.read()

    script = preprocess_for_word_counting(script)

    # load model
    if dataset == "all":
        with open("./models/statistical/imsdb/genres_word_counts.json", "r") as f:
            genres_word_counts = json.load(f)
        with open("./models/statistical/dailyscript/genres_word_counts.json", "r") as f:
            dailyscript_genres_word_counts = json.load(f)
        
        for genre in dailyscript_genres_word_counts:
            if genre in genres_word_counts:
                # add counts if word exists in both datasets or append if not
                for word in dailyscript_genres_word_counts[genre]:
                    if word in genres_word_counts[genre]:
                        genres_word_counts[genre][word] += dailyscript_genres_word_counts[genre][word]
                    else:
                        genres_word_counts[genre][word] = dailyscript_genres_word_counts[genre][word]
            else:
                genres_word_counts[genre] = dailyscript_genres_word_counts[genre]
    else:
        with open(f"./models/statistical/{dataset}/genres_word_counts.json", "r") as f:
            genres_word_counts = json.load(f)

    # predict
    counts = { genre: 0 for genre in genres_word_counts }
    for genre in genres_word_counts:
        for word in script:
            if word in genres_word_counts[genre]:
                counts[genre] += genres_word_counts[genre][word]

    prediction = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    prediction = list(map(lambda x: x[0], prediction[:number_of_classes]))
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="Name of the file with a movies script to be classified.")
    parser.add_argument("-n", "--number_of_classes", default=2, type=int, help="Number of classes to predict.")
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "all"],
                        help="Dataset to evaluate on.")
    args = parser.parse_args()
        
    prediction = predict(args.file_name, args.dataset, args.number_of_classes)
    print(f"Predicted genres: {prediction}")