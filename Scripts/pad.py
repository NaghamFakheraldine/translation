import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def pad_sequences_data():
    # Load dataset and tokenizers
    df = pd.read_csv("../data/translation_data.csv")

    with open('../tokenizers/tokenizer_en.pkl', 'rb') as f:
        eng_tokenizer = pickle.load(f)

    with open('../tokenizers/tokenizer_fr.pkl', 'rb') as f:
        fr_tokenizer = pickle.load(f)

    # Tokenize sentences
    eng_tokenized = eng_tokenizer.texts_to_sequences(df["English"])
    fr_tokenized = fr_tokenizer.texts_to_sequences(df["French"])

    max_english_length = max([len(seq) for seq in eng_tokenized])
    max_french_length = max([len(seq) for seq in fr_tokenized])

    eng_tokenized_padded = pad_sequences(eng_tokenized, maxlen=max_english_length, padding='post')
    fr_tokenized_padded = pad_sequences(fr_tokenized, maxlen=max_french_length, padding='post')

    return eng_tokenized_padded, fr_tokenized_padded, max_english_length, max_french_length

if __name__ == "__main__":
    eng_tokenized_padded, fr_tokenized_padded, max_english_length, max_french_length = pad_sequences_data()
    print(f"Max English Length: {max_english_length}")
    print(f"Max French Length: {max_french_length}")
