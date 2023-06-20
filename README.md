# Movie Genre Classification
This repository contains an implementation of a movie genre classification web application. The application apart from genre prediction also provides a GUI to:
* web-scrape necessary training data sets, 
* pre-process the data sets, 
* train selected model on a selected data set with selected movie genres,
* evaluate trained models on a test data set.

There are currently two available models, which perform the prediction, a statistical model and a machine learning model.

## Repository Structure
```
├── src                                       // source files of the project
│   ├── dataset_preparation                   // source file for pre-processing the scraped data sets  
│   │   ├── characters_cleaning.py
│   │   ├── data_prep_pipeline.sh
│   │   ├── description_cleaning.py
│   │   ├── html_tags_cleaning.py
│   │   ├── labels_cleaning.py
│   │   ├── merge.py
│   │   ├── names_cleaning.py
│   │   ├── pipeline.py
│   │   ├── punctuation_cleaning.py
│   │   ├── pure_text_separation.py
│   │   ├── single_character_cleaning.py
│   │   ├── special_cases_cleaning.py
│   │   ├── stemming.py
│   │   └── stopwords_removal.py
│   ├── scraping                              // source files for web-scraping of the data sets
│   │   ├── dailyscript.py
│   │   └── imsdb.py
│   ├── sentence_transformer_model            // source files of the machine learning model
│   │   ├── convert_all_embeddings.sh
│   │   ├── dataset_split.py
│   │   ├── embeddings.py
│   │   ├── evaluation.py
│   │   ├── genres.py
│   │   ├── labels.py
│   │   ├── prediction.py
│   │   └── training.py
│   ├── statistical_model                     // source files of the statistical model  
│   │   ├── dataset_split.py
│   │   ├── evaluation.py
│   │   ├── genres.py
│   │   ├── prediction.py
│   │   └── training.py
│   ├── visualization                         // source files of the web application
│   │   └── app.py
│   ├── seed.py
│   └── utils.py
├── batman_script.txt                         // test script for prediction
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Usage
First, install all necessary Python dependencies with:
```
pip install -r requirements.txt
```

Second, run the web application with:
```
python3 src/visualization/app.py
```
and open up the app at [http://127.0.0.1:8050](http://127.0.0.1:8050).

Third, the advised and fastest flow in the app is to scrape the `imsdb` data set, pre-process it, generate embeddings for the `average_word_embeddings_glove.6B.300d`, train, evaluate and predict using this model.

Last, since the web scraping and mainly generation of the embeddings can run even for several hours on HW without GPU support, we also provide all the data sets scraped and with generated embeddings for all the available models via this link [https://cloud.tugraz.at/index.php/s/ocgNRz5EwSt9A4k](https://cloud.tugraz.at/index.php/s/ocgNRz5EwSt9A4k). Extract the file into the root of the project.

## Web Scraping
### Imsdb scraper
This scraper extracts movie script data from the IMSDb website. 
It utilizes the Scrapy library to crawl and scrape the website. The `process_links()` function is a utility function that cleans the URLs of the links. The `ImsdbSpider` class is a custom spider that extends the `CrawlSpider` class from Scrapy. It specifies the target website, the starting URL, and the rules for following links. The class also includes methods for extracting genres and scripts from the scraped data. The `parse_item()` method is the callback function that handles the response and extracts relevant information. The `ImsdbScraper` class is responsible for running the scraping process and saving the scraped data to a JSON file.
Within the `parse_item()` method, after obtaining the response from the crawled URL, BeautifulSoup is used to parse the HTML content. It searches for the element that contains the genres information by finding the text "Genres" and navigating to its parent. From there, it finds all the `<a>` tags within that element. These links are then passed to the methods `_get_genres_from_links()` and `_get_script_from_links()` to extract the genres and script, respectively. The extracted title, genres, and script are yielded as a dictionary for each item scraped.


### Dailyscript scraper
The `DailyscriptScraper` class is designed to scrape movie scripts and genres from the "https://www.dailyscript.com/" website. It utilizes BeautifulSoup for HTML parsing and IMDb for genre information. The class includes various private methods, such as `_get_soup()` to retrieve webpage content, `_title_already_scraped()` to check if a movie has already been scraped, `_extract_movie_titles_and_script_links()` to obtain movie titles and script links, `_get_script()` to fetch movie scripts, `_get_genres_from_first_imdb_result()` to retrieve genres from IMDb, and `_save_data()` to store the scraped data in a JSON file. The main `run()` method orchestrates the scraping process, iterating through movies, checking their status, retrieving genres and scripts, and saving the data.


## Cleaning and Pre-processing
The web-scraped data are partial HTML pages still with HTML tags left. Furthermore, the scripts itself needs cleaning and pre-processing, as there are many special characters unnecessary spaces etc., which is accomplished by the following pipeline:
1. cleaning of special characters,
2. removal of HTML tags,
3. removal of punctuation characters,
4. removal of some special words like *CLOSE ON*, *CUT* etc.,
5. removal of names,
6. removal of descriptions of the scripts,
7. removal of stopwords (applied only for the statistical model),
8. stemming (applied only for the statistical model),
9. cleaning of the genre labels.

## Statistical Model
The statistical model is based on word counting and its primary role is to asses the performance of the machine learning model. Occurences of words are counted for each genre during training of the model. The prediction is then simply performed by looking up and summing the learned numbers of occurences of words in the predicted sample for all genres. The most probable genre or multiple genres is selected as the one with the largest sum(s). 

The performance of the statistical model trained on the `merged` data set and evaluated on 121 test samples is summarized below:
* Average IoU: 0.289
* Average recall: 0.394
* Average precision: 0.394
* Average F1 score: 0.394

## Machine Learning Model
The machine learning model is more complicated consisting of two sub-models. 

First, a pre-trained sentence transformer model available at [SBERT](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/). We currently support these models:
* `all-mpnet-base-v2`,
* `all-MiniLM-L12-v2`,
* `all-distilroberta-v1`,
* `multi-qa-mpnet-base-dot-v1`,
* `average_word_embeddings_glove.6B.300d`.
Unfortunately, we did not have the necessary compute resources to fine-tune any of these models to our data sets. The sentence transformer models are used to generate embeddings of the movie scripts splitted into samples with 256 words each, as this is the upper limit of a sentence length for some of the models. 

Second, a fully connected neural network with 3 hidden layers with ReLU activation function. The first and second hidden layers are also followed by a dropout layer with 0.25 dropout probability. The sigmoid activation function is used as the activation function of the output layer, since the model is trained to perform multi-class classification. The inputs of this model are the aforementioned embeddings generated by one of the sentence transformer models. The model is trained separately on each set of the generated embeddings with Adam optimizer with default parameters selected by the `TenserFlow` library. More precisely, we perform a 2 stage training. The model is first trained with an validation data set with an early stopping maximizing the validation accuracy with patience for 5 epochs as the stopping criterion. Then, the weights and biases are restored and the model is trained again on a training set including the validation set. The stopping criterion is selected as the epoch, where the validation training run reached the best accuracy.

The prediction is performed by spilling the movies script into samples with 256 words each, generating embeddings for each of the samples, predicting each sample. Finally, the predicted probabilities are averaged and the most probable genre or multiple genres are selected.

The performance of the machine learning models trained on the `merged` data set and evaluated on 121 test samples is summarized below (performance will vary depending on the weigh initialization as we are not using a fixed seed for better interactivity):
* `all-mpnet-base-v2`:
    * Average IoU: 0.467
    * Average recall: 0.562
    * Average precision: 0.657
    * Average F1 score: 0.572

* `all-MiniLM-L12-v2`:
    * Average IoU: 0.325
    * Average recall: 0.344
    * Average precision: 0.640
    * Average F1 score: 0.426

* `all-distilroberta-v1`:
    * Average IoU: 0.447
    * Average recall: 0.500
    * Average precision: 0.687
    * Average F1 score: 0.551

* `multi-qa-mpnet-base-dot-v1`:
    * Average IoU: 0.444
    * Average recall: 0.533
    * Average precision: 0.626
    * Average F1 score: 0.542

* `average_word_embeddings_glove.6B.300d`:
    * Average IoU: 0.390
    * Average recall: 0.434
    * Average precision: 0.676
    * Average F1 score: 0.498

## Web Application

The web application is implemented as a dash app. It consists of several tabs that provide different functionalities for data processing, model training, genre prediction, and model evaluation. Each tab serves a specific purpose and offers intuitive user interfaces for easy interaction.

### Data Sets Tab
In this tab, you can perform web scraping and data pre-processing tasks.  
For each step the previous step has to be executed at least once to work.

- **Web Scraping**: Select a data set from the dropdown menu and click the "Web Scrape" button to initiate the web scraping process. The scraped data will be displayed in a loading component.

- **Pre-process**: Choose a data set for pre-processing using the dropdown menu. Click the "Pre-process" button to start the pre-processing. The pre-processed output will be shown in a loading component.

- **Generate Embedding**: Select a data set and a model from the respective dropdown menus. Click the "Generate Embedding" button to generate embeddings for the chosen data set. The generated embeddings will be displayed in a loading component.

### Train Tab
In this tab, you can train and retrain models.

- **Training**: Choose a model and a data set from the dropdown menus. Additionally, select the desired genres for training (multi-select is available). Click the "Train" button to begin the training process. The progress will be shown in a loading component.

- **Retrain**: Select a trained model from the dropdown menu and click the "Retrain" button to initiate the retraining process.

### Prediction Tab
In this tab, you can predict genres based on trained models.

- **Predict Genre**: Select a trained model from the dropdown menu. Enter the number of genres to predict and provide a script in the textarea. Click the "Predict" button to generate genre predictions. The predictions will be displayed in a loading component.

### Evaluation Tab
In this tab, you can evaluate trained models and compare their performance.

- **Model Evaluation**: Choose a trained model and a data set from the respective dropdown menus. Click the "Evaluate" button to evaluate the selected model's performance. The evaluation results will be shown in a loading component.

- **Model Comparison**: This section provides a visual comparison of models
