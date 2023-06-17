import os
import sys
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils
import sentence_transformer_model.training as transformer_training
import sentence_transformer_model.prediction as transformer_prediction
import statistical_model.training as statistical_training
import statistical_model.prediction as statistical_prediction

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# Import necessary libraries for training, predicting, and evaluating the model

# Sample datasets and models
# TODO: change value to real models/datasets
model_dropdown_options = [
    {"label": "all-mpnet-base-v2", "value": "all-mpnet-base-v2"},
    {"label": "all-MiniLM-L12-v2", "value": "all-MiniLM-L12-v2"},
    {"label": "all-distilroberta-v1", "value": "all-distilroberta-v1"},
    {"label": "multi-qa-mpnet-base-dot-v1", "value": "multi-qa-mpnet-base-dot-v1"},
    {"label": "average_word_embeddings_glove.6B.300d", "value": "average_word_embeddings_glove.6B.300d"},
    {"label": "statistical", "value": "statistical"},
]
trained_model_dropdown_options = [ ]
dataset_dropdown_options = [
    "imsdb",
    "dailyscript",
    "merged",
]

# TODO: change value to real genres
available_genres = [
    {"label": "All", "value": "All"},
]

# Define app layout
app.layout = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            value="tab-train",
            children=[
                dcc.Tab(
                    label="Train",
                    value="tab-train",
                    children=[
                        html.Div(children=[
                            html.H3(
                                "Train Model",
                                style={"margin-top": "10px"},
                            ),
                            dcc.Dropdown(
                                id="dataset-dropdown",
                                placeholder="Select a dataset...",
                                style={
                                    "margin-top": "10px",
                                },
                                options=dataset_dropdown_options,
                            ),
                            dcc.Dropdown(
                                id="model-dropdown",
                                placeholder="Select a model...",
                                options=model_dropdown_options,
                                style={
                                    "margin-top": "10px",
                                },
                            ),
                            dcc.Dropdown(
                                id="train-genre-dropdown",
                                placeholder="Select the genres...",
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
                                "Retrain Model",
                                style={"margin-top": "10px"},
                            ),
                            dcc.Dropdown(
                                id="retrain-model-dropdown",
                                placeholder="Select a trained model...",
                                style={
                                    "margin-top": "10px",
                                },
                                options=trained_model_dropdown_options,
                                optionHeight=80,
                                maxHeight=500
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
                            )
                        ],
                        style={"width": "50%", "margin": "auto"}),
                    ],
                ),
                dcc.Tab(
                    label="Predict",
                    value="tab-predict",
                    children=[
                        html.Div(children=[
                            html.H3(
                                "Predict Genre",
                                style={"margin-top": "10px"},
                            ),
                            dcc.Dropdown(
                                id="trained-model-dropdown",
                                placeholder="Select a trained model...",
                                style={
                                    "margin-top": "10px",
                                },
                                options=trained_model_dropdown_options,
                                optionHeight=80,
                                maxHeight=500
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
                                children=[html.Div(id="predict-output", style={"margin-top": "10px"})],
                            ),
                        ],
                        style={"width": "50%", "margin": "auto"}),
                    ],
                ),
                dcc.Tab(
                    label="Evaluate",
                    value="tab-evaluate",
                    children=[
                        html.H3("Model Evaluation"),
                        dcc.Dropdown(
                            id="evaluating-model-dropdown",
                            placeholder="Select a trained model...",
                            options=trained_model_dropdown_options,
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                                "height": "50px",
                            },
                        ),
                        dcc.Loading(
                            id="evaluating-loading",
                            children=[
                                dcc.Graph(
                                    id="training-curve",
                                    figure={"data": [{"x": [], "y": []}]},
                                    style={
                                        "margin-top": "10px",
                                        "width": "50%",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Add Data",
                    value="tab-add-data",
                    children=[
                        html.H3("Add Data"),
                        dcc.Input(
                            id="new-script-title-input",
                            placeholder="Enter a script Title...",
                            style={
                                "margin-top": "10px",
                                "width": "30%",
                                "height": "50",
                            },
                        ),
                        html.Br(),
                        dcc.Textarea(
                            id="new-script-textarea",
                            placeholder="Enter new a script...",
                            style={
                                "margin-top": "10px",
                                "width": "30%",
                                "height": "200px",
                            },
                        ),
                        # List of genres
                        dcc.Dropdown(
                            id="new-script-genre-dropdown",
                            placeholder="Select the genres...",
                            options=available_genres,
                            multi=True,
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                            },
                        ),
                        dcc.Dropdown(
                            id="new-script-dataset-dropdown",
                            placeholder="Select a dataset...",
                            options=dataset_dropdown_options,
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                            },
                        ),
                        html.Button(
                            "Submit",
                            id="submit-new-script-button",
                            style={
                                "margin-top": "10px",
                                "width": "30%",
                                "height": "50px",
                            },
                        ),
                        html.Div(id="submit-new-script-output"),
                    ],
                ),
            ],
        )
    ],
)


# Callback to enable and disable the training unless a model and dataset are selected
@app.callback(
    Output("train-button", "disabled"),
    [Input("dataset-dropdown", "value"), Input("model-dropdown", "value"), Input("train-genre-dropdown", "value")],
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
    [Input("script-textarea", "value"),Input("trained-model-dropdown", "value")],
)
def enable_predict(text, model):
    if text and model:
        return False
    return True


# Callback to enable submit button unless all fields are filled
@app.callback(
    Output("submit-new-script-button", "disabled"),
    [
        Input("new-script-title-input", "value"),
        Input("new-script-textarea", "value"),
        Input("new-script-genre-dropdown", "value"),
        Input("new-script-dataset-dropdown", "value"),
    ],
)
def enable_submit(title, text, genres, dataset):
    if title and text and genres and dataset:
        # TODO: Check if the title is unique
        return False
    return True

# Callback to submit a new script
@app.callback(
    Output("submit-new-script-output", "children"),
    Input("submit-new-script-button", "n_clicks"),
    [
        State("new-script-title-input", "value"),
        State("new-script-textarea", "value"),
        State("new-script-genre-dropdown", "value"),
        State("new-script-dataset-dropdown", "value"),
    ],
)
def submit_new_script(n_clicks, title, text, genres, dataset):
    # TODO: Add the new script to the dataset
    if n_clicks:
        return html.Div("Script submitted 🎉")


# Show the training curve when a model is selected
@app.callback(
    Output("training-curve", "figure"),
    [Input("evaluating-model-dropdown", "value")],
)
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
        Output("evaluating-model-dropdown", "options"),
        Output("retrain-model-dropdown", "options"),
    ],
    [Input("train-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("model-dropdown", "value"), State("train-genre-dropdown", "value")],
    prevent_initial_call=True,
)
def train_model(n_clicks, dataset, model, genres):
    if n_clicks is not None:
        if "All" in genres:
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
            statistical_training.train(model_config["dataset"], model_config["genres"])
        else:
            transformer_training.train(model_config["model"], model_config["genres"], model_config["dataset"])
        
        update_trained_models()
        return html.Div("Model retrained successfully!")

# Callback for predicting the genre
@app.callback(
    Output("predict-output", "children"),
    [Input("predict-button", "n_clicks")],
    [
        State("script-textarea", "value"),
        State("trained-model-dropdown", "value"),
        State("num-genres-input", "value")
    ],
)
def predict_genre(n_clicks, script, model_hash, num_genres):
    if n_clicks is not None and script:
        num_genres = int(num_genres) if num_genres else 3
        model_config = utils.get_model_config_from_hash(model_hash)
        if model_config["model"] == "statistical":
            prediction = statistical_prediction.predict_string(model_config["genres"], model_config["dataset"], script, num_genres)
        else:
            prediction = transformer_prediction.predict_string(model_config["model"], model_config["genres"], model_config["dataset"], script, num_genres)

        return html.Div(f'Predicted genres: {", ".join(prediction)}')

def update_trained_models():
    trained_model_dropdown_options.clear()
    for model in utils.available_models():
        trained_model_dropdown_options.append(
            {
                "label": f"{model[0]} - {model[1]} - {', '.join(model[3])}", 
                "value": model[2]
            }
        )

if __name__ == "__main__":
    for genre in utils.available_genres():
        available_genres.append({"label": genre, "value": genre})

    update_trained_models()
    app.run_server(debug=True)
