import json

with open("./data/statistical_model/test_dataset.json", "r") as f:
    test_dataset = json.load(f)

with open("./data/statistical_model/genres_word_counts.json", "r") as f:
    genres_word_counts = json.load(f)

results = []
total_accuracy = 0

for sample in test_dataset:
    counts = { genre: 0 for genre in genres_word_counts }
    for genre in genres_word_counts:
        for word in sample["script"].split():
            if word in genres_word_counts[genre]:
                counts[genre] += genres_word_counts[genre][word]

    ground_truth = set(genre for dirty_genre in sample["genre"] if dirty_genre.strip() != "" for genre in dirty_genre.strip().split("."))
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    counts = counts[:len(ground_truth)]

    prediction = set(count[0] for count in counts)
    matches = prediction & ground_truth
    accuracy = len(matches) / len(ground_truth)
    total_accuracy += accuracy
    results.append({ sample["title"]: { "predicted": list(prediction), "actually": list(ground_truth), "accuracy": accuracy } })
    print(f"{sample['title']}:", 
          f"  - predicted: {prediction}", 
          f"  - actually:  {ground_truth}", 
          f"  - accuracy:  {accuracy}", sep='\n')

print(f"\nTotal accuracy: {total_accuracy / len(test_dataset)}")

with open("./data/statistical_model/results.json", "w") as f:
    json.dump(results, f, indent=2)
