
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import preprocess_for_word_counting, hash_model_attributes

def predict_string(genres: list[str], dataset: str, script: str, number_of_classes: int = 2):
    # load genres
    with open(f"./data/datasets/genres.json", "r") as f:
        all_genres = json.load(f)

    if len(genres) == 0:
        genres = all_genres
    else:
        genres = sorted(list(set(map(lambda x: x.strip(), genres))))
        genres = [ genre for genre in genres if genre in all_genres ]
    
    path = f"./models/{hash_model_attributes('statistical', genres, dataset)}"

    script = preprocess_for_word_counting(script)

    with open(f"{path}/genres_word_counts.json", "r") as f:
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

def predict_file(genres: list[str], dataset: str, script_file_name: str, number_of_classes: int = 2):
    # load script
    with open(script_file_name, "r") as f:
        script = f.read()
    
    return predict_string(genres, dataset, script, number_of_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="Name of the file with a movies script to be classified.")
    parser.add_argument("-g", "--genres", nargs='+', type=str, default=[],
                        help="Genres to train on separated by comma, unknown genres will be removed. If empty, train on all available genres.")
    parser.add_argument("-n", "--number_of_classes", default=2, type=int, help="Number of classes to predict.")
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset to predict on.")
    args = parser.parse_args()
        
    prediction = predict_file(args.genres, args.dataset, args.file_name, args.number_of_classes)
    print(f"Predicted genres: {prediction}")