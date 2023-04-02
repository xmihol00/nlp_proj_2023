import json
import os
import re
from typing import Dict, List, Union

import nltk
from tqdm import tqdm

# Just our sample type
SampleT = Dict[str, Union[str, List[str]]]


class MovieScriptPreprocessor:
    def __init__(self, out_file: str) -> None:
        self.out_file = out_file
        self._stopwords = set(nltk.corpus.stopwords.words("english"))
        self._stemmer = nltk.stem.snowball.SnowballStemmer("english")

    # PRIVATE METHODS
    def _load_samples(self, filename: str) -> List[SampleT]:
        with open(filename, mode="r") as f:
            data = json.load(f)
        return data

    def _load_data(self, filename: str) -> List[SampleT]:
        try:
            return self._load_samples(filename)
        except FileNotFoundError:
            return []

    def _save_sample(self, sample: SampleT):
        if not os.path.exists(self.out_file):
            with open(self.out_file, mode="w") as f:
                json.dump([sample], f)
        else:
            with open(self.out_file, mode="r+") as f:
                data = json.load(f)
                data.append(sample)
                f.seek(0)
                json.dump(data, f)

    def _step_clean_characters(self, script: str) -> str:
        script = re.sub(r"\\'", "'", script)
        script = re.sub(r"((\")|(\\t)|(\[')|('\])|(--)|(\[)|(\]))", "", script)
        script = re.sub(r"((\s+)|(\\r\\n)+)", " ", script)
        script = re.sub(
            r"((,\s'\s',\s)|(\s',\s)|(\s'\s)|(\s,\s))", " ", script
        )
        script = re.sub(r"((\s')|('\s))", " ", script)
        # converting whole words in uppercase to lowercase
        script = re.sub(r"[A-Z]{2,}", lambda x: x.group().lower(), script)
        return script

    def _step_clean_html_tags(self, script: str) -> str:
        script = re.sub(
            (
                r"((<b>.*?</b>)|(<script>.*?</script>)|"
                r"(<html>)|(</html>)|(<title>.*?</title>)|"
                r"(<head>)|(</head>)|(<body.*?>)|"
                r"(</body>)|(<pre>)|(</pre>))"
            ),
            "",
            script,
            re.DOTALL,
        )
        script = re.sub(r"(<b>.*?</b>)", "", script)
        return script

    def _step_clean_punctuation(self, script: str) -> str:
        script = re.sub(
            r"(\.|,|\?|\!|\))\s+[A-Z]", lambda x: x.group().lower(), script
        )
        script = re.sub(
            (
                r"((\.+\s+)|(\s+\.+)|(\s*,\s+)|"
                r"(\s+')|('\s+)|(:\s+)|(;\s+)|"
                r"(\?)|(\!))"
            ),
            " ",
            script,
        )
        script = re.sub(r"((\.\.\.)|(\.\.)|(-\s)|(\s-))", " ", script)
        script = re.sub(r"(\.|,|:|;|-)", " ", script)
        script = re.sub(r"(\s+)", " ", script)
        return script

    def _step_clean_special_cases(self, script: str) -> str:
        script = re.sub(
            r"((CLOSE ON:)|(CUT:)|(ANGLE ON:)|(\(Cont.\)\s))", "", script
        )
        script = re.sub(r"(^\s+)", "", script)
        script = re.sub(r"(\d+)", "", script)
        script = re.sub(r"(\s+)", " ", script)
        script = re.sub(
            r"(\@|\#|\$|\%|\^|\&|\*|\(|\)|\{|\}|\]|\[|\<|\>|\_|\|\~|\+|/)",
            "",
            script,
        )
        script = re.sub(r"\s+", " ", script)
        script = re.sub(r" I ", " i ", script)
        return script

    def _step_pure_text_separation(self, script: str) -> str:
        return "" + script

    def _step_clean_description(self, script: str) -> str:
        script = script[150:]
        script += " "
        while script[0] != " ":
            script = script[1:]
        return script[1:-1]

    def _step_clean_stopwords(self, script: str) -> str:
        return " ".join(
            [word for word in script.split() if word not in self._stopwords]
        )

    def _step_stemming(self, script: str) -> str:
        return " ".join([self._stemmer.stem(word) for word in script.split()])

    def _apply_title_steps(self, title: str) -> str:
        return title.lower()

    def _apply_genres_steps(self, genres: List[str]) -> List[str]:
        return [g.lower() for g in genres]

    def _apply_script_steps(self, script: str) -> str:
        script = self._step_clean_characters(script)
        script = self._step_clean_html_tags(script)
        script = self._step_clean_punctuation(script)
        script = self._step_pure_text_separation(script)
        script = self._step_clean_description(script)
        script = self._step_clean_stopwords(script)
        script = self._step_stemming(script)
        return script

    def _apply_sample_steps(self, sample: SampleT) -> SampleT:
        sample["title"] = self._apply_title_steps(sample["title"])
        sample["genres"] = self._apply_genres_steps(sample["genres"])
        sample["script"] = self._apply_script_steps(sample["script"])
        return sample

    # PUBLIC METHODS
    def process_sample(self, sample: SampleT):
        new_sample = self._apply_sample_steps(sample)
        self._save_sample(new_sample)
        return new_sample

    def process_file(self, filename: str):
        samples = self._load_data(filename)
        out_data = self._load_data(self.out_file)
        out_titles = [s["title"].lower() for s in out_data]
        for sample in tqdm(samples, desc=f"Preprocessing for {filename}"):
            if sample["title"].lower() in out_titles:
                continue
            self.process_sample(sample)

    def load_output(self) -> List[SampleT]:
        return self._load_samples(self.out_file)
