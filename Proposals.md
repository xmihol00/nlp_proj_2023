# Text Catgorization
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

# Trending Posts
* **Large text**: Extract post from different subreddits/twitter accounts and parse into schema
* **Pipelines**: 
	1. Find sentiment/emotion of a post + comment section (maybe find out what was the trigger) 
	2. ** Entity recognition **: Parse result in a schema like 
	```
	subredditId: str,
	userId: str,
	postId: str,
	parentPostID: postId (Null if new post)
	postText: str,
	isComment: bool,
	upvotes: int
	datetime: int (utc)
	sentiment: float,
	emotion: enum or str,
	trigger: str,
	```
	3. Train model on that schema, that will predict if a post will trend.
* **Interactive output**: Make GUI that can take a text as input and a subreddit and tell the user the propability that this post will trend. A trending post would be one with many upvotes.
* **Anaysis**: Find out keywords that may trigger a trending post and cluster based on trigger and subreddit for example.
