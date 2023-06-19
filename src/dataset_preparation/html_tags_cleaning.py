import argparse
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="imsdb", choices=["imsdb", "dailyscript"],
                    help="Dataset to process.")
args = parser.parse_args()

print("Cleaning HTML tags...")

with open(f"./data/datasets/{args.dataset}/characters_cleaned.json", "r") as f:
    data = json.load(f)

for sample in data:
    sample["script"] = re.sub(r"((<b>.*?</b>)|(<script>.*?</script>)|(<html>)|(</html>)|(<title>.*?</title>)|" + 
                              r"(<head>)|(</head>)|(<body.*?>)|(</body>)|(<pre>)|(</pre>))", "", sample["script"], re.DOTALL)
    sample["script"] = re.sub(r"(<b>.*?</b>)", "", sample["script"])

with open(f"./data/datasets/{args.dataset}/html_tags_cleaned.json", "w") as f:
    json.dump(data, f, indent=2)
