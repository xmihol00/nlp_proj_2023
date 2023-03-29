import re


print(re.sub(r"\.\s+[A-Z]", lambda x: x.group().lower(), "Come and join us. She smiles a crazy smile. Reaches out for Quaid. A SNAKE appears from around the back of her"))
