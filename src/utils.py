import hashlib
import re
import nltk
from nltk.stem.snowball import SnowballStemmer

def hash_model_name_and_labels(model_name, labels):
    input_str = model_name + "".join(sorted(set(labels)))
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