import re
import json

with open("./data/datasets/characters_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"((<b>.*?</b>)|(<script>.*?</script>)|(<html>)|(</html>)|(<title>.*?</title>)|" + 
                              r"(<head>)|(</head>)|(<body.*?>)|(</body>)|(<pre>)|(</pre>))", "", sample["script"], re.DOTALL)
    sample["script"] = re.sub(r"(<b>.*?</b>)", "", sample["script"])

with open("./data/datasets/html_tags_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)
