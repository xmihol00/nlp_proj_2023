# Privacy-Preservation on Text
* **Large text**: Use of multiple data sets.
* **Pipelines**:
    1. ML: Data sets download, cleaning, pre-processing, training, evaluation, visualisation of results.
    2. Inference: Own document upload, pre-processing, inference on the trained ML model, presentation of results.
* **Pre-processing**: Conversion of all the data sets to the same schema, i.e. consistent word separation and labeling.
* **Analysis**: Summarization of how our solution performs via graphs or tables, clustering of test documents based on types of private information.
* **Interactive output analysis**: GUI for document uploads, which enable the user to further add pseudonyms or remove the detected by the inference pipeline.

# Text Catgorization
Some more specific ideas:
1.  Categorizing Wikipedia articles based on a field of studies, e.g. history - https://en.wikipedia.org/wiki/Index_of_history_articles economics - https://en.wikipedia.org/wiki/Index_of_economics_articles, chemistry - https://en.wikipedia.org/wiki/Index_of_chemistry_articles, ...
1. Categorizing based on an origin of a text, e.g. Wikipedia articles, scientific papers, blog posts, Facebook posts, tweets, ...
1. Detecting movie genre based on its script. The scripts would be web crawled, e.g. from these domains https://imsdb.com and https://www.simplyscripts.com.
## Project outline
* **Large text**: The sources named above.
* **Pipelines**:
    1. ML: Web crawling of data sets, cleaning, pre-processing (e.g. splitting into smaller chunks and labeling), training, evaluation, visualization of results.
    2. Inference: Own document upload, pre-processing (e.g. conversion to the format needed for inference), inference on the trained ML model, presentation of results.
* **Pre-processing**: Conversion of all the crawled data sets to the same schema, i.e. consistent word separation and labeling.
* **Analysis**: Summarization of how our solution performs via graphs, tables or a confusion matrix, clustering of test documents based on their category.
* **Interactive output analysis**: GUI for uploading of documents, which will enable the user to select specific parts of the text to be categorized from each document.
