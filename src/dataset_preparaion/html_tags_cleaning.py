import re
import json

with open("./data/characters_cleaned_imsdb_data.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"(<b>.*?</b>)|(<script>.*?</script>)|(<html>)|(</html>)|(<title>.*?</title>)|" + 
                              r"(<body.*?>)|(</body>)|(<pre>)|(</pre>)", "", sample["script"], re.DOTALL)

with open("./data/html_tags_cleaned_imsdb_data.json", "w") as f:
    json.dump(data, f, indent=2)