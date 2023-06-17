import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import hash_model_attributes

def train(dataset: str, genres: list[str], normalize: bool = False):
    with open(f"./data/statistical_model/{dataset}/train_dataset.json", "r") as f:
        data = json.load(f)
    
    with open(f"./data/datasets/genres.json", "r") as f:
        all_genres = json.load(f)
    
    model_name = "statistical"
    if len(genres) == 0:
        genres_indices = list(range(len(all_genres)))
        dir_name = hash_model_attributes(model_name, all_genres, dataset)
    else:
        picked_genres = sorted(list(set(map(lambda x: x.strip(), genres))))
        picked_genres = [ genre for genre in picked_genres if genre in all_genres ]
        genres_indices = [ all_genres.index(genre) for genre in picked_genres ]
        dir_name = hash_model_attributes(model_name, picked_genres, dataset)
        all_genres = picked_genres

    config = {}
    config["genres"] = all_genres
    config["genres_indices"] = genres_indices
    config["model"] = model_name
    config["hash"] = dir_name
    config["dataset"] = dataset

    genres_word_counts = { genre: {} for genre in all_genres }
    genres_occurences = { genre: 0 for genre in all_genres }

    for sample in data:
        for dirty_genre in sample["genres"]:
            if dirty_genre.strip() != "":
                for genre in dirty_genre.strip().split("."):
                    if genre in genres_word_counts:
                        genres_occurences[genre] += 1
                        for word in sample["script"].split():
                            if word in genres_word_counts[genre]:
                                genres_word_counts[genre][word] += 1
                            else:
                                genres_word_counts[genre][word] = 1

    if normalize:
        max_genre_counts = 0
        for genre in genres_word_counts:
            max_genre_counts = max(max_genre_counts, len(genres_word_counts[genre]))

        for genre in genres_word_counts:
            sum = 0
            for word in genres_word_counts[genre]:
                sum += genres_word_counts[genre][word]
            sum -= genres_occurences[genre] * 100
            
            for word in genres_word_counts[genre]:
                genres_word_counts[genre][word] /= sum

    os.makedirs(f"./models/{dir_name}", exist_ok=True)
    with open(f"./models/{dir_name}/genres_word_counts.json", "w") as f:
        json.dump(genres_word_counts, f, indent=2)
    with open(f"./models/{dir_name}/config.json", "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--genres", nargs='+', type=str, default=[],
                        help="Genres to train on separated by comma, unknown genres will be removed. If empty, train on all available genres.")
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript", "merged"],
                        help="Dataset to train on.")
    args = parser.parse_args()
    train(args.dataset, args.genres)