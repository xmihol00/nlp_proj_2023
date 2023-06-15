import time

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# Import necessary libraries for training, predicting, and evaluating the model

# Sample datasets and models
datasets = ["Dataset 1", "Dataset 2", "Dataset 3"]
models = ["Model 1", "Model 2", "Model 3"]
trained_models = ["Trained Model 1", "Trained Model 2", "Trained Model 3"]
all_generes = [
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
                            style={"margin-top": "10px", "allign": "center"},
                        ),
                        dcc.Dropdown(
                            id="dataset-dropdown",
                            placeholder="Select a dataset...",
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                                "height": "50px",
                            },
                            options=[
                                {"label": dataset, "value": dataset}
                                for dataset in datasets
                            ],
                        ),
                        dcc.Dropdown(
                            id="model-dropdown",
                            placeholder="Select a model...",
                            options=[
                                {"label": model, "value": model}
                                for model in models
                            ],
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                                "height": "50px",
                            },
                        ),
                        html.Button(
                            "Train",
                            id="train-button",
                            style={
                                "margin-top": "10px",
                                "width": "30%",
                                "height": "50px",
                            },
                        ),
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
                            style={"margin-top": "10px", "allign": "center"},
                        ),
                        dcc.Dropdown(
                            id="trainde-model-dropdown",
                            placeholder="Select a trained model...",
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                                "height": "50px",
                            },
                            options=[
                                {"label": model, "value": model}
                                for model in trained_models
                            ],
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
                            options=[
                                {"label": model, "value": model}
                                for model in trained_models
                            ],
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
                        )
                        # Add components to show training curve and accuracy
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
                        html.Textarea(
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
                                for genre in all_generes
                            ],
                            multi=True,
                            style={
                                "margin-top": "10px",
                                "width": "50%",
                            },
                        ),
                    ],
                ),
            ],
        )
    ]
)


# Callback to enable and disable the training unless a model and dataset are selected
@app.callback(
    Output("train-button", "disabled"),
    [Input("dataset-dropdown", "value"), Input("model-dropdown", "value")],
)
def enable_train(dataset, model):
    if dataset and model:
        return False
    return True


# Callback to enable and disable the training unless a model and dataset are selected
@app.callback(
    Output("predict-button", "disabled"),
    [
        Input("script-textarea", "value"),
        Input("trainde-model-dropdown", "value"),
    ],
)
def enable_predict(text, model):
    if text and model:
        return False
    return True

# Show the training curve when a model is selected
@app.callback(
    Output("training-curve", "figure"), [Input("evaluating-model-dropdown", "value")]
)
def show_training_curve(model):
    # Demo x and y values
    time.sleep(2)
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 2, 1]
    if model:
        # Load the model from the trained_models folder
        # Plot the training curve
        return {"data": [{"x": x, "y": y}]}
    return {"data": [{"x": [], "y": []}]}


# Callback for training the model
@app.callback(
    Output("train-output", "children"),
    [Input("train-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("model-dropdown", "value")],
)
def train_model(n_clicks, dataset, model):
    if n_clicks is not None:
        # Perform the model training based on the selected dataset and model
        # Update the train-output div with the training results
        # Write a small message to the user that this may take a while
        time.sleep(5)  # Simulate a long running prediction
        return html.Div("Model trained successfully!")


# Callback for predicting the genre
@app.callback(
    Output("predict-output", "children"),
    [Input("predict-button", "n_clicks")],
    [State("script-textarea", "value")],
)
def predict_genre(n_clicks, script):
    if n_clicks is not None and script:
        # Perform the prediction based on the entered script
        # Update the predict-output div with the predicted genres
        genres = [
            "Genre 1",
            "Genre 2",
            "Genre 3",
        ]  # Sample genres, replace with actual prediction
        return html.Div(f'Predicted genres: {", ".join(genres)}')


if __name__ == "__main__":
    app.run_server(debug=True)
