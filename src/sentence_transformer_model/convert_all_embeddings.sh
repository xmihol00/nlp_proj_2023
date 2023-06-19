python3 src/sentence_transformer_model/dataset_split.py
python3 src/sentence_transformer_model/labels.py
python3 src/sentence_transformer_model/embeddings.py -m all-mpnet-base-v2
python3 src/sentence_transformer_model/embeddings.py -m all-MiniLM-L12-v2
python3 src/sentence_transformer_model/embeddings.py -m all-distilroberta-v1
python3 src/sentence_transformer_model/embeddings.py -m multi-qa-mpnet-base-dot-v1
python3 src/sentence_transformer_model/embeddings.py -m average_word_embeddings_glove.6B.300d