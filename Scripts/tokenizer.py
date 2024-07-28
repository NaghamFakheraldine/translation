import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import os

DATA_FILE = "../data/translation_data.csv"
TOKENIZER_EN_FILE = "../tokenizers/tokenizer_en.pkl"
TOKENIZER_FR_FILE = "../tokenizers/tokenizer_fr.pkl"

def tokenize_sentences():
    df = pd.read_csv(DATA_FILE)

    eng_tokenizer = Tokenizer()
    eng_tokenizer.fit_on_texts(df["English"])

    fr_tokenizer = Tokenizer()
    fr_tokenizer.fit_on_texts(df["French"])

    # Save tokenizers
    os.makedirs(os.path.dirname(TOKENIZER_EN_FILE), exist_ok=True)
    with open(TOKENIZER_EN_FILE, 'wb') as f:
        pickle.dump(eng_tokenizer, f)
    
    with open(TOKENIZER_FR_FILE, 'wb') as f:
        pickle.dump(fr_tokenizer, f)

if __name__ == "__main__":
    tokenize_sentences()
