import pandas as pd
from prediction import predict_string
import plotly.express as px
import json

df = pd.read_csv("sentiment/tweets.csv")[["airline_sentiment", "text"]]

positive_sentiment = df[df["airline_sentiment"] == "positive"]
negative_sentiment = df[df["airline_sentiment"] == "negative"]
neutral_sentiment = df[df["airline_sentiment"] == "neutral"]

# print counts of samples for each sentiment
print("positive sentiment:", len(positive_sentiment), "samples")
print("negative sentiment:", len(negative_sentiment), "samples")
print("neutral sentiment:", len(neutral_sentiment), "samples")

# reduce the number of samples for each sentiment to 1000
positive_sentiment = positive_sentiment.sample(n=1000)
negative_sentiment = negative_sentiment.sample(n=1000)
neutral_sentiment = neutral_sentiment.sample(n=1000)

model = "multi-qa-mpnet-base-dot-v1"
# predict genres for each sentiment
positive_sentiment_predictions = []
for _, row in positive_sentiment.iterrows():
    positive_sentiment_predictions += predict_string(model, [], "merged", row["text"])

negative_sentiment_predictions = []
for _, row in negative_sentiment.iterrows():
    negative_sentiment_predictions += predict_string(model, [], "merged", row["text"])

neutral_sentiment_predictions = []
for _, row in neutral_sentiment.iterrows():
    neutral_sentiment_predictions += predict_string(model, [], "merged", row["text"])

# collect counts of genres for each sentiment
with open(f"./data/datasets/genres.json", "r") as f:
    all_genres = json.load(f)

rows = []
columns = ["sentiment", "genre", "count"]
for genre in all_genres:
    rows.append(["positive", genre, positive_sentiment_predictions.count(genre)])
    rows.append(["negative", genre, negative_sentiment_predictions.count(genre)])
    rows.append(["neutral", genre, neutral_sentiment_predictions.count(genre)])

# save counts of genres for each sentiment to csv and plot
df = pd.DataFrame(rows, columns=columns)
df.to_csv(f"sentiment/predicted_genres_{model}.csv", index=False)
fig = px.bar(df, x="sentiment", y="count", color="genre", barmode="group")
fig.show()
