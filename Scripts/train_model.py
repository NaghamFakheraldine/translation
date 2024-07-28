import pickle
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.utils import to_categorical
from pad import pad_sequences_data

def train_model():
    eng_tokenized_padded, fr_tokenized_padded, max_english_length, max_french_length = pad_sequences_data()

    with open('../tokenizers/tokenizer_en.pkl', 'rb') as f:
        eng_tokenizer = pickle.load(f)

    with open('../tokenizers/tokenizer_fr.pkl', 'rb') as f:
        fr_tokenizer = pickle.load(f)

    eng_vocab_size = len(eng_tokenizer.word_index) + 1 
    fr_vocab_size = len(fr_tokenizer.word_index) + 1

    fr_tokenized_padded_encoded = to_categorical(fr_tokenized_padded, num_classes=fr_vocab_size)

    model = Sequential()
    model.add(Embedding(input_dim=eng_vocab_size, output_dim=100, input_length=max_english_length))
    model.add(Bidirectional(GRU(20)))
    model.add(RepeatVector(max_french_length))
    model.add(Bidirectional(GRU(20, return_sequences=True)))
    model.add(TimeDistributed(Dense(fr_vocab_size, activation='softmax')))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(eng_tokenized_padded, fr_tokenized_padded_encoded, validation_split=0.2, epochs=25, batch_size=64)

    if not os.path.exists('../model'):
        os.makedirs('../model')

    model.save('../model/model.h5')

if __name__ == "__main__":
    train_model()
