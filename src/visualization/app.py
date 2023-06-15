import time

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# Import necessary libraries for training, predicting, and evaluating the model

# Sample datasets and models
# TODO: change value to real models/datasets
model_dropdown_options = [
    {"label": "Model 1", "value": "Model 1"},
    {"label": "Model 2", "value": "Model 2"},
    {"label": "Model 3", "value": "Model 3"},
]
trained_model_dropdown_options = [
    {"label": "Trained Model 1", "value": "Trained Model {1"},
    {"label": "Trained Model 2", "value": "Trained Model 2"},
    {"label": "Trained Model 3", "value": "Trained Model 3"},
]
dataset_dropdown_options = [
    {"label": "Dataset 1", "value": "Dataset 1"},
    {"label": "Dataset 2", "value": "Dataset 2"},
    {"label": "Dataset 3", "value": "Dataset 3"},
]

# TODO: change value to real genres
available_genres = [
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Horror",
    "Romance",
    "Thriller",
]

# Define app layout
app.layout = html.Center(
    [
        dcc.Tabs(
            id="tabs",
            value="tab-train",
            children=[
                dcc.Tab(
                    label="Train",
                    value="tab-train",
                    children=[
                        html.H3(
                            "Train Model",
                            style={"margin-top": "10px"},
                        ),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            placeholder="Select a dataset...",
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                                "height": "50px",
                            },
                            options=dataset_dropdown_options,
                        ),
                        dcc.Dropdown(
                            id="model-dropdown",
                            placeholder="Select a model...",
                            options=model_dropdown_options,
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                                "height": "50px",
                            },
                        ),
                        html.Div(
                            children=[
                                html.Button(
                                    "Train New",
                                    id="train-button",
                                    style={
                                        "margin-top": "10px",
                                        "width": "50%",
                                        "height": "50px",
                                    },
                                ),
                                html.Button(
                                    "Retrain",
                                    id="retrain-button",
                                    style={
                                        "margin-top": "10px",
                                        "width": "50%",
                                        "height": "50px",
                                    },
                                ),
                            ],
                            style={
                                "margin-top": "10px",
                                "width": "30%",
                                "height": "50px",
                            },
                        ),
                        html.Br(),
                        dcc.Loading(
                            id="training-loading",
                            children=[html.Div(id="train-output")],
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Predict",
                    value="tab-predict",
                    children=[
                        html.H3(
                            "Predict Genre",
                            style={"margin-top": "10px"},
                        ),
                        dcc.Dropdown(
                            id="trained-model-dropdown",
                            placeholder="Select a trained model...",
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                                "height": "50px",
                            },
                            options=trained_model_dropdown_options,
                        ),
                        dcc.Textarea(
                            id="script-textarea",
                            placeholder="Enter a script...",
                            style={
                                "margin-top": "10px",
                                "width": "30%",
                                "height": "200px",
                            },
                        ),
                        html.Br(),
                        html.Button(
                            "Predict",
                            id="predict-button",
                            style={
                                "margin-top": "10px",
                                "width": "30%",
                                "height": "50px",
                            },
                        ),
                        dcc.Loading(
                            id="predict-loading",
                            children=[html.Div(id="predict-output")],
                        ),
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
                            options=[
                                {"label": genre, "value": genre}
                                for genre in available_genres
                            ],
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
    ]
)


# Callback to enable and disable the training unless a model and dataset are selected
@app.callback(
    [Output("train-button", "disabled"), Output("retrain-button", "disabled")],
    [Input("dataset-dropdown", "value"), Input("model-dropdown", "value")],
)
def enable_train(dataset, model):
    if dataset and model:
        return False, False
    return True, True


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
    ],
    [Input("train-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("model-dropdown", "value")],
    prevent_initial_call=True,
)
def train_model(n_clicks, dataset, model):
    if n_clicks is not None:
        time.sleep(5)  # Simulate a long running prediction
        # TODO: Perform the model training based on the selected dataset and model
        # TODO: train new model here
        new_model_name = f"New Model {time.time()}"
        new_model_options = {"label": new_model_name, "value": new_model_name}
        trained_model_dropdown_options.append(new_model_options)
        return (
            html.Div("Model trained successfully!"),
            trained_model_dropdown_options,
            trained_model_dropdown_options,
        )


@app.callback(
    [
        Output("train-output", "children", allow_duplicate=True),
        Output("trained-model-dropdown", "options", allow_duplicate=True),
        Output("evaluating-model-dropdown", "options", allow_duplicate=True),
    ],
    [Input("retrain-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("model-dropdown", "value")],
    prevent_initial_call=True,
)
def retrain_model(n_clicks, dataset, model):
    if n_clicks is not None:
        # TODO: Perform the model retraining based on the selected dataset and model
        # Replace the model name with the new model
        new_model = model
        model_name = model
        for i, m in enumerate(trained_model_dropdown_options):
            if m["label"] == model_name:
                trained_model_dropdown_options[i] = {
                    "label": model_name,
                    "value": new_model,
                }
        time.sleep(2)  # Simulate a long running training
        return (
            html.Div("Model retrained successfully!"),
            trained_model_dropdown_options,
            trained_model_dropdown_options,
        )


# Callback for predicting the genre
@app.callback(
    Output("predict-output", "children"),
    [Input("predict-button", "n_clicks")],
    [
        State("script-textarea", "value"),
        State("trained-model-dropdown", "value"),
    ],
)
def predict_genre(n_clicks, script, model):
    if n_clicks is not None and script:
        # Perform the prediction based on the entered script
        # Update the predict-output div with the predicted genres

        # TODO: Perform the model prediction based on the selected model and script
        genres = [
            "Genre 1",
            "Genre 2",
            "Genre 3",
        ]  # Sample genres, replace with actual prediction
        return html.Div(f'Predicted genres: {", ".join(genres)}')


if __name__ == "__main__":
    app.run_server(debug=True)
