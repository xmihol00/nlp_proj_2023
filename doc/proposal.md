# Text categorization of movie scripts
## Project Description:

The project involves building a baseline model to predict the genre of a given movie script and a proper model that utilizes the transformer architecture.
The baseline model will calculate the number of occurrences of words appearing in each genre, remove stop words, and perform stemming.
The proper model will use an encoder from the transformer architecture, including an embedding layer, multi-head attention layers, and a MLP to predict the genre.
The majority of the project's work will be spent on data crawling, cleaning, labeling, preprocessing and building a pipeline.

After the prediction is performed on the dataset clustering is perfomed to see which words indicate that a given movie script belongs to a genre.
A GUI/TUI should be provided for a user to categorize own text snippets/scripts

## Goal:

The goal of this project is to build a robust NLP text categorization model that accurately predicts the genre of a given movie script.
The project aims to achieve high accuracy while minimizing the number of false positives and false negatives.
A GUI/TUI interface will be provided to enable a user to categorize text snippets.
Get insights from clustering to make asumptions on trigger words for specific movie genres.



## Dataset:

The project will use a dataset of crawled and cleaned movie scripts that have been labeled with their corresponding genres.
The dataset will be preprocessed by removing stop words and performing stemming.
The size of the dataset will depend on the desired accuracy of the model.
The dataset will be split into training, validation, and test sets.

Sources:
- imsdb.com
- simplyscripts.com

## Risk Analysis and Backup Plan:

There are several risks associated with this project, including:

**Lack of data:** If the dataset is not large enough, the model may not achieve the desired accuracy.
The backup plan for this risk is to augment the dataset by using data augmentation techniques such as synonym replacement, random insertion, and random deletion.

**Overfitting:** If the model overfits to the training data, it may perform poorly on unseen data.
The backup plan for this risk is to use regularization techniques such as dropout, weight decay, and early stopping.

**Model complexity:** If the transformer model is too complex, it may take too long to train and may not generalize well. 
The backup plan for this risk is to use a simpler model architecture or to use pre-trained transformer models such as BERT or GPT.

**Technical issues:** There may be technical issues with the data preprocessing, model training, or evaluation.
