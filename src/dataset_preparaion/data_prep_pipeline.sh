
python3 src/dataset_preparaion/characters_cleaning.py        -d $1
python3 src/dataset_preparaion/html_tags_cleaning.py         -d $1
python3 src/dataset_preparaion/punctuation_cleaning.py       -d $1
python3 src/dataset_preparaion/special_cases_cleaning.py     -d $1
python3 src/dataset_preparaion/names_cleaning.py             -d $1  
python3 src/dataset_preparaion/description_cleaning.py       -d $1
python3 src/dataset_preparaion/stopwards_removal.py          -d $1 
python3 src/dataset_preparaion/stemming.py                   -d $1
python3 src/dataset_preparaion/single_character_cleaning.py  -d $1
python3 src/dataset_preparaion/labels_cleaning.py            -d $1
 