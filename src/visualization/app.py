import os
import sys
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Import necessary libraries for training, predicting, and evaluating the model
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils
import sentence_transformer_model.training as transformer_training
import sentence_transformer_model.prediction as transformer_prediction
import sentence_transformer_model.evaluation as transformer_evaluation
import sentence_transformer_model.embeddings as transformer_embeddings
import sentence_transformer_model.labels as transformer_labels
import sentence_transformer_model.dataset_split as transformer_dataset_split
import sentence_transformer_model.genres as transformer_genres
import statistical_model.training as statistical_training
import statistical_model.prediction as statistical_prediction
import statistical_model.evaluation as statistical_evaluation
import statistical_model.dataset_split as statistical_dataset_split
import statistical_model.genres as statistical_genres
import dataset_preparation.merge as merge
from scraping.dailyscript import DailyscriptScraper
from scraping.imsdb import ImsdbScraper

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

models = [
    "all-mpnet-base-v2",
    "all-MiniLM-L12-v2",
    "all-distilroberta-v1",
    "multi-qa-mpnet-base-dot-v1",
    "average_word_embeddings_glove.6B.300d",
]

# models
model_dropdown_options = [{"label": model, "value": model} for model in models]
models_embeddings_dropdown_options = []

# datasets
web_dataset_dropdown_options = [
    "imsdb",
    "dailyscript",
]
dataset_dropdown_options = [
    "imsdb",
    "dailyscript",
    "merged",
]
scraped_dataset_dropdown_options = []
preprocessed_dataset_dropdown_options = []
datasets_with_embeddings = {
    "imsdb": [],
    "dailyscript": [],
    "merged": [],
}
dataset_with_embeddings_dropdown_options = []

trained_model_dropdown_options = []
available_genres = [
    {"label": "All", "value": "all"},
]

compared_models = []

# Define app layout
app.layout = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            value="tab-data-sets",
            children=[
                dcc.Tab(
                    label="Data Sets",
                    value="tab-data-sets",
                    children=[
                        html.Div(
                            children=[
                                html.H3(
                                    "Web Scrape", style={"margin-top": "10px"}
                                ),
                                dcc.Dropdown(
                                    id="web-scrape-dataset-dropdown",
                                    placeholder="Select a data set to be web scraped...",
                                    options=web_dataset_dropdown_options,
                                ),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Web Scrape",
                                            id="web-scrape-button",
                                            style={
                                                "margin-top": "10px",
                                                "height": "50px",
                                                "width": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "25%",
                                        "height": "50px",
                                        "margin": "auto",
                                    },
                                ),
                                dcc.Loading(
                                    id="web-scraping-loading",
                                    children=[
                                        html.Div(
                                            id="web-scraping-output",
                                            style={"margin-top": "10px"},
                                        )
                                    ],
                                ),
                                html.H3(
                                    "Pre-process",
                                    style={"padding-top": "10px"},
                                ),
                                dcc.Dropdown(
                                    id="preprocessed-dataset-dropdown",
                                    placeholder="Select a data set for pre-processing...",
                                    options=scraped_dataset_dropdown_options,
                                ),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Pre-process",
                                            id="pre-process-button",
                                            style={
                                                "margin-top": "10px",
                                                "height": "50px",
                                                "width": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "25%",
                                        "height": "50px",
                                        "margin": "auto",
                                    },
                                ),
                                dcc.Loading(
                                    id="pre-processing-loading",
                                    children=[
                                        html.Div(
                                            id="pre-processing-output",
                                            style={"margin-top": "10px"},
                                        )
                                    ],
                                ),
                                html.H3(
                                    "Generate Embedding",
                                    style={"padding-top": "10px"},
                                ),
                                dcc.Dropdown(
                                    id="embeddings-dataset-dropdown",
                                    placeholder="Select a data set to generate embeddings for...",
                                    options=preprocessed_dataset_dropdown_options,
                                ),
                                dcc.Dropdown(
                                    id="embeddings-model-dropdown",
                                    placeholder="Select a model to generate embeddings for...",
                                    options=model_dropdown_options,
                                    style={"margin-top": "10px"},
                                ),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Generate Embedding",
                                            id="embeddings-button",
                                            style={
                                                "margin-top": "10px",
                                                "height": "50px",
                                                "width": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "25%",
                                        "height": "50px",
                                        "margin": "auto",
                                    },
                                ),
                                dcc.Loading(
                                    id="embeddings-loading",
                                    children=[
                                        html.Div(
                                            id="embeddings-output",
                                            style={"margin-top": "10px"},
                                        )
                                    ],
                                ),
                            ],
                            style={"width": "50%", "margin": "auto"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Train",
                    value="tab-train",
                    children=[
                        html.Div(
                            children=[
                                html.H3(
                                    "Training",
                                    style={"margin-top": "10px"},
                                ),
                                dcc.Dropdown(
                                    id="model-dropdown",
                                    placeholder="Select a model...",
                                    options=models_embeddings_dropdown_options,
                                ),
                                dcc.Dropdown(
                                    id="dataset-dropdown",
                                    placeholder="Select a dataset...",
                                    options=dataset_with_embeddings_dropdown_options,
                                    style={
                                        "margin-top": "10px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="train-genre-dropdown",
                                    placeholder="Select genres...",
                                    options=available_genres,
                                    multi=True,
                                    style={
                                        "margin-top": "10px",
                                    },
                                ),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Train",
                                            id="train-button",
                                            style={
                                                "margin-top": "10px",
                                                "height": "50px",
                                                "width": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "25%",
                                        "height": "50px",
                                        "margin": "auto",
                                    },
                                ),
                                html.H3(
                                    "Retrain",
                                    style={"padding-top": "10px"},
                                ),
                                dcc.Dropdown(
                                    id="retrain-model-dropdown",
                                    placeholder="Select a trained model...",
                                    options=trained_model_dropdown_options,
                                    optionHeight=80,
                                    maxHeight=500,
                                ),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Retrain",
                                            id="retrain-button",
                                            style={
                                                "margin-top": "10px",
                                                "height": "50px",
                                                "width": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "25%",
                                        "height": "50px",
                                        "margin": "auto",
                                    },
                                ),
                                html.Br(),
                                dcc.Loading(
                                    id="training-loading",
                                    children=[html.Div(id="train-output")],
                                ),
                            ],
                            style={"width": "50%", "margin": "auto"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Prediction",
                    value="tab-predict",
                    children=[
                        html.Div(
                            children=[
                                html.H3(
                                    "Predict Genre",
                                    style={"margin-top": "10px"},
                                ),
                                dcc.Dropdown(
                                    id="trained-model-dropdown",
                                    placeholder="Select a trained model...",
                                    options=trained_model_dropdown_options,
                                    optionHeight=80,
                                    maxHeight=500,
                                ),
                                dcc.Input(
                                    id="num-genres-input",
                                    type="number",
                                    placeholder="Enter a number of genres to predict...",
                                    style={
                                        "margin-top": "10px",
                                        "width": "100%",
                                    },
                                ),
                                dcc.Textarea(
                                    id="script-textarea",
                                    placeholder="Enter a script...",
                                    style={
                                        "margin-top": "10px",
                                        "height": "200px",
                                        "width": "100%",
                                    },
                                ),
                                html.Br(),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Predict",
                                            id="predict-button",
                                            style={
                                                "margin-top": "10px",
                                                "height": "50px",
                                                "width": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "25%",
                                        "height": "50px",
                                        "margin": "auto",
                                    },
                                ),
                                dcc.Loading(
                                    id="predict-loading",
                                    children=[
                                        html.Div(
                                            id="predict-output",
                                            style={"margin-top": "10px"},
                                        )
                                    ],
                                ),
                            ],
                            style={"width": "50%", "margin": "auto"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Evaluation",
                    value="tab-evaluate",
                    children=[
                        html.Div(
                            children=[
                                html.H3(
                                    "Model Evaluation",
                                    style={"margin-top": "10px"},
                                ),
                                dcc.Dropdown(
                                    id="evaluation-model-dropdown",
                                    placeholder="Select a trained model...",
                                    options=trained_model_dropdown_options,
                                    optionHeight=80,
                                    maxHeight=500,
                                ),
                                dcc.Dropdown(
                                    id="evaluation-dataset-dropdown",
                                    placeholder="Select a data set to evaluate on...",
                                    options=preprocessed_dataset_dropdown_options,
                                    style={
                                        "margin-top": "10px",
                                    },
                                ),
                                html.Div(
                                    children=[
                                        html.Button(
                                            "Evaluate",
                                            id="evaluate-button",
                                            style={
                                                "margin-top": "10px",
                                                "height": "50px",
                                                "width": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "25%",
                                        "height": "50px",
                                        "margin": "auto",
                                    },
                                ),
                                dcc.Loading(
                                    id="evaluating-loading",
                                    children=[
                                        html.Div(
                                            id="evaluation-output",
                                            style={"margin-top": "10px"},
                                        ),
                                        html.Ul(id="evaluation-results"),
                                    ],
                                ),
                                html.H3(
                                    "Model Comparison",
                                    style={"padding-top": "10px"},
                                ),
                                html.Div(
                                    id="model-comparison",
                                    children=compared_models,  # TODO: replace with a nice plot
                                ),
                            ],
                            style={"width": "50%", "margin": "auto"},
                        ),
                    ],
                ),
                # TODO: move the functionality to Data sets tab
                # dcc.Tab(
                #    label="Add Data",
                #    value="tab-add-data",
                #    children=[
                #        html.H3("Add Data"),
                #        dcc.Input(
                #            id="new-script-title-input",
                #            placeholder="Enter a script Title...",
                #            style={
                #                "margin-top": "10px",
                #                "width": "30%",
                #                "height": "50",
                #            },
                #        ),
                #        html.Br(),
                #        dcc.Textarea(
                #            id="new-script-textarea",
                #            placeholder="Enter new a script...",
                #            style={
                #                "margin-top": "10px",
                #                "width": "30%",
                #                "height": "200px",
                #            },
                #        ),
                #        # List of genres
                #        dcc.Dropdown(
                #            id="new-script-genre-dropdown",
                #            placeholder="Select the genres...",
                #            options=available_genres,
                #            multi=True,
                #            style={
                #                "margin-top": "10px",
                #                "width": "50%",
                #            },
                #        ),
                #        dcc.Dropdown(
                #            id="new-script-dataset-dropdown",
                #            placeholder="Select a dataset...",
                #            options=dataset_dropdown_options,
                #            style={
                #                "margin-top": "10px",
                #                "width": "50%",
                #            },
                #        ),
                #        html.Button(
                #            "Submit",
                #            id="submit-new-script-button",
                #            style={
                #                "margin-top": "10px",
                #                "width": "30%",
                #                "height": "50px",
                #            },
                #        ),
                #        html.Div(id="submit-new-script-output"),
                #    ],
                # ),
            ],
        )
    ],
)


# Callback to enable and disable the training unless a model and dataset are selected
@app.callback(
    Output("train-button", "disabled"),
    [
        Input("dataset-dropdown", "value"),
        Input("model-dropdown", "value"),
        Input("train-genre-dropdown", "value"),
    ],
)
def enable_train(dataset, model, genre):
    if dataset and model and genre:
        return False
    return True


# Callback to enable and disable the retraining unless a model is selected
@app.callback(
    Output("retrain-button", "disabled"),
    [Input("retrain-model-dropdown", "value")],
)
def enable_train(model):
    if model:
        return False
    return True


# Callback to enable and disable the training unless a model and dataset are selected
@app.callback(
    Output("predict-button", "disabled"),
    [
        Input("script-textarea", "value"),
        Input("trained-model-dropdown", "value"),
    ],
)
def enable_predict(text, model):
    if text and model:
        return False
    return True


# Callback to enable and disable the evaluation unless a model and dataset are selected
@app.callback(
    Output("evaluate-button", "disabled"),
    [
        Input("evaluation-model-dropdown", "value"),
        Input("evaluation-dataset-dropdown", "value"),
    ],
)
def enable_evaluate(model, dataset):
    if model and dataset:
        return False
    return True


# Callback to enable and disable the evaluation unless a model and dataset are selected
@app.callback(
    Output("web-scrape-button", "disabled"),
    [Input("web-scrape-dataset-dropdown", "value")],
)
def enable_webscraping(dataset):
    if dataset:
        return False
    return True


@app.callback(
    Output("pre-process-button", "disabled"),
    [Input("preprocessed-dataset-dropdown", "value")],
)
def enable_preprocess(dataset):
    if dataset:
        return False
    return True


@app.callback(
    Output("embeddings-button", "disabled"),
    [
        Input("embeddings-dataset-dropdown", "value"),
        Input("embeddings-model-dropdown", "value"),
    ],
)
def enable_embeddings(dataset, model):
    if dataset and model:
        return False
    return True


# TODO: Callback to enable submit button unless all fields are filled
# @app.callback(
#    Output("submit-new-script-button", "disabled"),
#    [
#        Input("new-script-title-input", "value"),
#        Input("new-script-textarea", "value"),
#        Input("new-script-genre-dropdown", "value"),
#        Input("new-script-dataset-dropdown", "value"),
#    ],
# )
def enable_submit(title, text, genres, dataset):
    if title and text and genres and dataset:
        # TODO: Check if the title is unique
        return False
    return True


# TODO: Callback to submit a new script
# @app.callback(
#    Output("submit-new-script-output", "children"),
#    Input("submit-new-script-button", "n_clicks"),
#    [
#        State("new-script-title-input", "value"),
#        State("new-script-textarea", "value"),
#        State("new-script-genre-dropdown", "value"),
#        State("new-script-dataset-dropdown", "value"),
#    ],
# )
def submit_new_script(n_clicks, title, text, genres, dataset):
    # TODO: - Pre-process the script
    #       - Add the new script to the selected dataset
    #       - Create embeddings for the new script for all transformer models
    #       - Create labels for the new script for the transformer models (y_train_labels.npy)
    #       - Append the new embeddings to the X_train_*.npy files
    #       - Append the script also to the statistical model (train_dataset.json)
    if n_clicks:
        return html.Div("Script submitted 🎉")


# NTH: It would be nice to show the training curve as the model is trained, but this is not a priority. It might be quite difficult to implement.
# @app.callback(
#    Output("training-curve", "figure"),
#    [Input("evaluation-model-dropdown", "value")],
# )
def show_training_curve(model):
    # Demo x and y values
    time.sleep(2)
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 2, 1]
    if model:
        # TODO: Plot the training curve for the selected model
        return {"data": [{"x": x, "y": y}]}
    return {"data": [{"x": [], "y": []}]}


# Callback for training the model
@app.callback(
    [
        Output("train-output", "children"),
        Output("trained-model-dropdown", "options"),
        Output("evaluation-model-dropdown", "options"),
        Output("retrain-model-dropdown", "options"),
    ],
    [Input("train-button", "n_clicks")],
    [
        State("dataset-dropdown", "value"),
        State("model-dropdown", "value"),
        State("train-genre-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def train_model(n_clicks, dataset, model, genres):
    if n_clicks is not None:
        if "all" in genres:
            genres = []

        if model == "statistical":
            statistical_training.train(genres, dataset)
        else:
            transformer_training.train(model, genres, dataset)

        update_trained_models()
        return (
            html.Div("Model trained successfully!"),
            trained_model_dropdown_options,
            trained_model_dropdown_options,
            trained_model_dropdown_options,
        )


@app.callback(
    Output("train-output", "children", allow_duplicate=True),
    [Input("retrain-button", "n_clicks")],
    [State("retrain-model-dropdown", "value")],
    prevent_initial_call=True,
)
def retrain_model(n_clicks, model_hash):
    if n_clicks is not None:
        model_config = utils.get_model_config_from_hash(model_hash)
        if model_config["model"] == "statistical":
            statistical_training.train(
                model_config["dataset"], model_config["genres"]
            )
        else:
            transformer_training.train(
                model_config["model"],
                model_config["genres"],
                model_config["dataset"],
            )

        update_trained_models()
        return html.Div("Model retrained successfully!")


# Callback for predicting the genre
@app.callback(
    Output("predict-output", "children"),
    [Input("predict-button", "n_clicks")],
    [
        State("script-textarea", "value"),
        State("trained-model-dropdown", "value"),
        State("num-genres-input", "value"),
    ],
)
def predict_genre(n_clicks, script, model_hash, num_genres):
    if n_clicks is not None and script:
        num_genres = int(num_genres) if num_genres else 3
        model_config = utils.get_model_config_from_hash(model_hash)
        if model_config["model"] == "statistical":
            prediction = statistical_prediction.predict_string(
                model_config["genres"],
                model_config["dataset"],
                script,
                num_genres,
            )
        else:
            prediction = transformer_prediction.predict_string(
                model_config["model"],
                model_config["genres"],
                model_config["dataset"],
                script,
                num_genres,
            )

        return html.Div(f'Predicted genres: {", ".join(prediction)}')


@app.callback(
    [
        Output("evaluation-output", "children"),
        Output("evaluation-results", "children"),
    ],
    [Input("evaluate-button", "n_clicks")],
    [
        State("evaluation-model-dropdown", "value"),
        State("evaluation-dataset-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def evaluate_model(n_clicks, model_hash, dataset):
    if n_clicks is not None:
        model_config = utils.get_model_config_from_hash(model_hash)
        if model_config["model"] == "statistical":
            total_predicted, average_metrics = statistical_evaluation.evaluate(
                model_config["genres"], model_config["dataset"], dataset
            )
        else:
            total_predicted, average_metrics = transformer_evaluation.evaluate(
                model_config["model"],
                model_config["genres"],
                model_config["dataset"],
                dataset,
            )

        return (
            html.Div(f"Model evaluated on {total_predicted} samples:"),
            [
                html.Li(f"{metric}: {value}")
                for metric, value in filter(
                    lambda x: x[0].startswith("Average"),
                    average_metrics.items(),
                )
            ],
        )


@app.callback(
    [
        Output("pre-processing-output", "children"),
        Output("embeddings-dataset-dropdown", "options"),
        Output("train-genre-dropdown", "options"),
    ],
    [Input("pre-process-button", "n_clicks")],
    [State("preprocessed-dataset-dropdown", "value")],
    prevent_initial_call=True,
)
def preprocess_dataset(n_clicks, dataset):
    if n_clicks is not None:
        if dataset == "merged":
            os.system(f"./src/dataset_preparation/data_prep_pipeline.sh imsdb")
            os.system(
                f"./src/dataset_preparation/data_prep_pipeline.sh dailyscript"
            )
            transformer_genres.extract_genres()
            statistical_genres.extract_genres()
            transformer_dataset_split.split_dataset("merged")
            statistical_dataset_split.split_dataset("merged")
            merge.merge_datasets()
            transformer_labels.encode_labels("merged")
        else:
            os.system(
                f"./src/dataset_preparation/data_prep_pipeline.sh {dataset}"
            )
            transformer_genres.extract_genres()
            statistical_genres.extract_genres()
            transformer_dataset_split.split_dataset(dataset)
            statistical_dataset_split.split_dataset(dataset)
            transformer_labels.encode_labels(dataset)

        update_preprocessed_datasets()
        update_available_genres()
        return (
            html.Div("Dataset preprocessed successfully!"),
            preprocessed_dataset_dropdown_options,
            available_genres,
        )


@app.callback(
    [
        Output("embeddings-output", "children"),
        Output("model-dropdown", "options"),
    ],
    [Input("embeddings-button", "n_clicks")],
    [
        State("embeddings-model-dropdown", "value"),
        State("embeddings-dataset-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def generate_embeddings(n_clicks, model, dataset):
    if n_clicks is not None:
        transformer_embeddings.generate_embeddings(
            model, f"./data/sentence_transformer_model/{dataset}"
        )
        update_models_with_embeddings()
        return (
            html.Div("Embeddings generated successfully!"),
            models_embeddings_dropdown_options,
        )


@app.callback(
    [
        Output("web-scraping-output", "children"),
        Output("preprocessed-dataset-dropdown", "options"),
    ],
    [Input("web-scrape-button", "n_clicks")],
    [State("web-scrape-dataset-dropdown", "value")],
    prevent_initial_call=True,
)
def webscrape_dataset(n_clicks, dataset):
    if n_clicks is not None:
        if dataset == "imsdb":
            os.system(f"python3 ./src/scraping/imsdb.py")
        elif dataset == "dailyscript":
            os.system(f"python3 ./src/scraping/dailyscript.py")

        update_scraped_datasets()
        return (
            html.Div("Dataset web scraped successfully!"),
            scraped_dataset_dropdown_options,
        )

# Callback that triggers if a model was selected to be trainend that modifies the
# Datset dropdown to only show datasets that have embeddings for that model
@app.callback(
    Output("dataset-dropdown", "options"),
    [Input("model-dropdown", "value")],
)
def update_datasets_with_embeddings(model):
    dataset_with_embeddings_dropdown_options = []
    for dataset in datasets_with_embeddings:
        if model in datasets_with_embeddings[dataset]:
            dataset_with_embeddings_dropdown_options.append(
                {"label": dataset, "value": dataset}
            )
    return dataset_with_embeddings_dropdown_options

def update_models_with_embeddings():
    models_embeddings_dropdown_options.clear()
    models_embeddings_dropdown_options.append(
        {"label": "statistical", "value": "statistical"}
    )
    for dataset in dataset_dropdown_options:
        datasets_with_embeddings[dataset].clear()

    for model in models:
        has_embeddings = False
        for dataset in dataset_dropdown_options:
            if os.path.exists(
                f"./data/sentence_transformer_model/{dataset}/X_test_embeddings_{model}.npy"
            ) and os.path.exists(
                f"./data/sentence_transformer_model/{dataset}/X_train_embeddings_{model}.npy"
            ):
                has_embeddings = True
                datasets_with_embeddings[dataset].append(model)

        if has_embeddings:
            models_embeddings_dropdown_options.append(
                {"label": model, "value": model}
            )

def update_trained_models():
    trained_model_dropdown_options.clear()
    for model in utils.available_models():
        trained_model_dropdown_options.append(
            {
                "label": f"{model[0]} - {model[1]} - {', '.join(model[3])}",
                "value": model[2],
            }
        )


def update_scraped_datasets():
    scraped_dataset_dropdown_options.clear()
    if os.path.exists("./data/scraped_data/scraped_imsdb_data.json"):
        scraped_dataset_dropdown_options.append(
            {"label": "imsdb", "value": "imsdb"}
        )
    if os.path.exists("./data/scraped_data/scraped_dailyscript_data.json"):
        scraped_dataset_dropdown_options.append(
            {"label": "dailyscript", "value": "dailyscript"}
        )
    if len(scraped_dataset_dropdown_options) == 2:
        scraped_dataset_dropdown_options.append(
            {"label": "merged", "value": "merged"}
        )


def update_preprocessed_datasets():
    preprocessed_dataset_dropdown_options.clear()
    for dataset in dataset_dropdown_options:
        path_prefix = "./data/sentence_transformer_model/"
        if (
            os.path.exists(f"{path_prefix}{dataset}/test_dataset_whole_scripts.json")
            and os.path.exists(f"{path_prefix}{dataset}/test_dataset.json")
            and os.path.exists(f"{path_prefix}{dataset}/train_dataset.json")
        ):
            preprocessed_dataset_dropdown_options.append(
                {"label": dataset, "value": dataset}
            )

def update_available_genres():
    available_genres.clear()
    for genre in utils.available_genres():
        available_genres.append({"label": genre, "value": genre})

    if len(available_genres) > 0:
        available_genres.insert(0, {"label": "all", "value": "all"})


def update_compared_models():
    compared_models.clear()
    for evaluation in utils.models_with_metrics():
        model, dataset, _, genres, metric_dataset_dict = evaluation
        for evaluation_dataset, metrics in metric_dataset_dict.items():
            compared_models.append(
                html.Div(children=[
                    # TODO: assign a color and shape to each model
                    html.Span(f"{model} - {dataset} - {', '.join(genres)}", style={"font-weight": "bold"}),
                    html.Span(f" evaluated on {evaluation_dataset} data set with {metrics['samples']} samples:"),
                    # TODO: instead of lists, create plots for each metric comparing the models
                    html.Ul(children=
                        [html.Li(f"{metric}: {value}") for metric, value in filter(lambda x: x[0].startswith('Average'), metrics.items())]
                    )
                ])
            )


if __name__ == "__main__":
    update_available_genres()
    update_scraped_datasets()
    update_preprocessed_datasets()
    update_models_with_embeddings()
    update_trained_models()
    update_compared_models()
    app.run_server(debug=True)
