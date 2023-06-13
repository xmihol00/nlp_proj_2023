import argparse
import json

def train(dataset: str, normalize: bool = False):
    with open(f"./data/statistical_model/{dataset}/train_dataset.json", "r") as f:
        data = json.load(f)
    
    with open(f"./data/statistical_model/{dataset}/genres.json", "r") as f:
        genres = json.load(f)

    genres_word_counts = { genre: {} for sample in data for dirty_genre in sample["genre"] if dirty_genre.strip() != "" for genre in dirty_genre.strip().split(".") }
    genres_occurences = { genre: 0 for sample in data for dirty_genre in sample["genre"] if dirty_genre.strip() != "" for genre in dirty_genre.strip().split(".") }

    for sample in data:
        for dirty_genre in sample["genre"]:
            if dirty_genre.strip() != "":
                for genre in dirty_genre.strip().split("."):
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

    with open(f"./models/statistical/{dataset}/genres_word_counts.json", "w") as f:
        json.dump(genres_word_counts, f, indent=2)
    
    with open(f"./models/statistical/{dataset}/genres.json", "w") as f:
        json.dump(genres, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                        help="Dataset to evaluate on.")
    args = parser.parse_args()
    train(args.dataset)