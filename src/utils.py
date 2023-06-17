import hashlib
import json
import os
import re
import nltk
from nltk.stem.snowball import SnowballStemmer

def hash_model_attributes(model_name: str, labels: list[str], dataset: str):
    input_str = model_name + "".join(sorted(set(labels))) + dataset
    return hashlib.md5(input_str.encode()).hexdigest()

def preprocess_for_embedding(script):
    script = re.sub(r"\\'", "'", script)
    script = re.sub(r"((\")|(\\t)|(\[')|('\])|(--)|(\[)|(\]))", "", script)
    script = re.sub(r"((\s+)|(\\r\\n)+)", " ", script)
    script = re.sub(r"((,\s'\s',\s)|(\s',\s)|(\s'\s)|(\s,\s))", " ", script)
    script = re.sub(r"((\s')|('\s))", " ", script)
    script = re.sub(r"[A-Z]{2,}", lambda x: x.group().lower(), script)
    script = re.sub(r"((<b>.*?</b>)|(<script>.*?</script>)|(<html>)|(</html>)|(<title>.*?</title>)|" + 
                    r"(<head>)|(</head>)|(<body.*?>)|(</body>)|(<pre>)|(</pre>))", "", script, re.DOTALL)
    script = re.sub(r"(<b>.*?</b>)", "", script)
    script = re.sub(r"(\.|,|\?|\!|\))\s+[A-Z]", lambda x: x.group().lower(), script) 
    script = re.sub(r"((\.+\s+)|(\s+\.+)|(\s*,\s+)|(\s+')|('\s+)|(:\s+)|(;\s+)|(\?)|(\!))", " ", script)
    script = re.sub(r"((\.\.\.)|(\.\.)|(-\s)|(\s-))", " ", script)
    script = re.sub(r"(\.|,|:|;|-)", " ", script)
    script = re.sub(r"(\s+)", " ", script)
    script = re.sub(r"((CLOSE ON:)|(CUT:)|(ANGLE ON:)|(\(Cont.\)\s))", "", script)
    script = re.sub(r"(^\s+)", "", script)
    script = re.sub(r"(\d+)", "", script)
    script = re.sub(r"(\s+)", " ", script)
    script = re.sub(r"(\@|\#|\$|\%|\^|\&|\*|\(|\)|\{|\}|\]|\[|\<|\>|\_|\|\~|\+|/)", "", script)
    script = re.sub(r"\s+", " ", script)
    script = re.sub(r" I ", " i ", script)
    script = re.sub(r"[A-Z]\w*", "", script)
    script = re.sub(r"'s", "", script)
    script = re.sub(r"\s+", " ", script)
    return script.strip()

def preprocess_for_word_counting(script):
    script = re.sub(r"\\'", "'", script)
    script = re.sub(r"((\")|(\\t)|(\[')|('\])|(--)|(\[)|(\]))", "", script)
    script = re.sub(r"((\s+)|(\\r\\n)+)", " ", script)
    script = re.sub(r"((,\s'\s',\s)|(\s',\s)|(\s'\s)|(\s,\s))", " ", script)
    script = re.sub(r"((\s')|('\s))", " ", script)
    script = re.sub(r"[A-Z]{2,}", lambda x: x.group().lower(), script)
    script = re.sub(r"((<b>.*?</b>)|(<script>.*?</script>)|(<html>)|(</html>)|(<title>.*?</title>)|" + 
                    r"(<head>)|(</head>)|(<body.*?>)|(</body>)|(<pre>)|(</pre>))", "", script, re.DOTALL)
    script = re.sub(r"(<b>.*?</b>)", "", script)
    script = re.sub(r"(\.|,|\?|\!|\))\s+[A-Z]", lambda x: x.group().lower(), script) 
    script = re.sub(r"((\.+\s+)|(\s+\.+)|(\s*,\s+)|(\s+')|('\s+)|(:\s+)|(;\s+)|(\?)|(\!))", " ", script)
    script = re.sub(r"((\.\.\.)|(\.\.)|(-\s)|(\s-))", " ", script)
    script = re.sub(r"(\.|,|:|;|-)", " ", script)
    script = re.sub(r"(\s+)", " ", script)
    script = re.sub(r"((CLOSE ON:)|(CUT:)|(ANGLE ON:)|(\(Cont.\)\s))", "", script)
    script = re.sub(r"(^\s+)", "", script)
    script = re.sub(r"(\d+)", "", script)
    script = re.sub(r"(\s+)", " ", script)
    script = re.sub(r"(\@|\#|\$|\%|\^|\&|\*|\(|\)|\{|\}|\]|\[|\<|\>|\_|\|\~|\+|/)", "", script)
    script = re.sub(r"\s+", " ", script)
    script = re.sub(r" I ", " i ", script)
    script = re.sub(r"[A-Z]\w*", "", script)
    script = re.sub(r"'s", "", script)
    script = re.sub(r"\s+", " ", script)

    stopwords = set(nltk.corpus.stopwords.words("english"))
    script = " ".join([word for word in script.split() if word not in stopwords])
    
    stemmer = SnowballStemmer("english")
    script = " ".join([stemmer.stem(word) for word in script.split()])
    
    script = re.sub(r"\s.\s", " ", script)

    return script.strip()

def available_models() -> list[tuple[str, str, str|None, list[str]]]:
    """
    Searches for all available models.
    :return: list of tuples (model name, dataset name, hash, genres)
    """

    models = []
    for model_dir in os.listdir("./models"):
        with open(os.path.join("./models", model_dir, "config.json")) as f:
            config = json.load(f)
        models.append((config["model"], config["dataset"], config["hash"], config["genres"]))
        
    return models

def model_exists(model_name: str, dataset: str, genres: list[str]) -> bool:
    """
    :return: True if the model exists, False otherwise
    """
    return os.path.exists(os.path.join("./models", hash_model_attributes(model_name, genres, dataset)))

def get_model_metrics(model_name: str, dataset: str, genres: list[str]) -> dict[str, float]:
    """
    :return: dictionary of model metrics
    :throws: FileNotFoundError if evaluation metrics file does not exist
    """

    with open(os.path.join("./models", hash_model_attributes(model_name, genres, dataset), "metrics.json")) as f:
        return json.load(f)

def get_model_metrics_from_hash(model_hash: str) -> dict[str, float]:
    """
    :return: dictionary of model metrics
    :throws: FileNotFoundError if evaluation metrics file does not exist
    """

    with open(os.path.join("./models", model_hash, "metrics.json")) as f:
        return json.load(f)

def get_model_config(model_name: str, dataset: str, genres: list[str]) -> dict:
    """
    :return: dictionary of model config
    """
    with open(os.path.join("./models", hash_model_attributes(model_name, genres, dataset), "config.json")) as f:
        return json.load(f)

def get_model_config_from_hash(model_hash: str) -> dict:
    """
    :return: dictionary of model config
    """
    with open(os.path.join("./models", model_hash, "config.json")) as f:
        return json.load(f)

def models_with_metrics() -> list[tuple[str, str, str|None, list[str]]]:
    """
    :return: list of tuples (model name, dataset name, hash, genres) with available metrics
    """

    models = []
    for model_dir in os.listdir("./models"):
        if os.path.exists(os.path.join("./models", model_dir, "metrics.json")):
            with open(os.path.join("./models", model_dir, "config.json")) as f:
                config = json.load(f)
            models.append((config["model"], config["dataset"], config["hash"], config["genres"]))
        
    return models

def has_model_metrics(model_name: str, dataset: str, genres: list[str]) -> bool:
    """
    :return: True if the model has metrics, False otherwise
    """

    return os.path.exists(os.path.join("./models", hash_model_attributes(model_name, genres, dataset), "metrics.json"))

def available_genres():
    with open("data/datasets/genres.json") as f:
        return json.load(f)

if __name__ == "__main__":
    print("Available models:")
    for model in available_models():
        print(model)
    
    print("\nModels with metrics:")
    for model in models_with_metrics():
        print(model)
