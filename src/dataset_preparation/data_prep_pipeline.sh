
python3 src/dataset_preparation/characters_cleaning.py        -d $1
python3 src/dataset_preparation/html_tags_cleaning.py         -d $1
python3 src/dataset_preparation/punctuation_cleaning.py       -d $1
python3 src/dataset_preparation/special_cases_cleaning.py     -d $1
python3 src/dataset_preparation/names_cleaning.py             -d $1  
python3 src/dataset_preparation/description_cleaning.py       -d $1
python3 src/dataset_preparation/stopwords_removal.py          -d $1 
python3 src/dataset_preparation/stemming.py                   -d $1
python3 src/dataset_preparation/single_character_cleaning.py  -d $1
python3 src/dataset_preparation/labels_cleaning.py            -d $1
 