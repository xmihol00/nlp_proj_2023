import os
import json


def extract_genres(all_datasets: list[str] = ["imsdb", "dailyscript"]):
    os.makedirs("./data/sentence_transformer_model/", exist_ok=True)

    genres = set()
    for dataset in all_datasets:
        with open(f"./data/datasets/{dataset}/final.json", "r") as f:
            data = json.load(f)

        # get all genres to use as labels
        for sample in data:
            genres.update(sample["genres"])
    
    with open(f"./data/datasets/genres.json", "w") as f:
        json.dump(sorted(list(genres)), f, indent=2)

def available_genres():
    with open("./data/datasets/genres.json", "r") as f:
        return json.load(f)

if __name__ == "__main__":
    extract_genres()