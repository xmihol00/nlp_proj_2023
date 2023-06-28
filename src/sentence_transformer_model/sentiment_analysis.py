import pandas as pd
from prediction import predict_string
import plotly.express as px
import json

df = pd.read_csv("sentiment/tweets.csv")[["airline_sentiment", "text"]]

positive_sentiment = df[df["airline_sentiment"] == "positive"]
negative_sentiment = df[df["airline_sentiment"] == "negative"]
neutral_sentiment = df[df["airline_sentiment"] == "neutral"]

positive_sentiment_predictions = []
for _, row in positive_sentiment.iterrows():
    positive_sentiment_predictions += predict_string("all-mpnet-base-v2", [], "merged", row["text"])
    break

negative_sentiment_predictions = []
for _, row in negative_sentiment.iterrows():
    negative_sentiment_predictions += predict_string("all-mpnet-base-v2", [], "merged", row["text"])
    break

neutral_sentiment_predictions = []
for _, row in neutral_sentiment.iterrows():
    neutral_sentiment_predictions += predict_string("all-mpnet-base-v2", [], "merged", row["text"])
    break

with open(f"./data/datasets/genres.json", "r") as f:
    all_genres = json.load(f)

rows = []
columns = ["sentiment", "genre", "count"]
for genre in all_genres:
    rows.append(["positive", genre, positive_sentiment_predictions.count(genre)])
    rows.append(["negative", genre, negative_sentiment_predictions.count(genre)])
    rows.append(["neutral", genre, neutral_sentiment_predictions.count(genre)])

df = pd.DataFrame(rows, columns=columns)
df.to_csv("sentiment/predicted_genres.csv", index=False)
fig = px.bar(df, x="sentiment", y="count", color="genre", barmode="group")
fig.show()
