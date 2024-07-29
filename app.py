import gradio as gr
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizers and model
with open('tokenizers/tokenizer_en.pkl', 'rb') as f:
    eng_tokenizer = pickle.load(f)
with open('tokenizers/tokenizer_fr.pkl', 'rb') as f:
    fr_tokenizer = pickle.load(f)
model = load_model('model/model.h5')

max_english_length = 15
max_french_length = 21

def translate_sentence(input_sentence):
    input_seq = eng_tokenizer.texts_to_sequences([input_sentence])
    input_seq_padded = pad_sequences(input_seq, maxlen=max_english_length, padding='post')

    prediction = model.predict(input_seq_padded)
    predicted_sentence = ' '.join([fr_tokenizer.index_word[np.argmax(word)] 
                                   if np.argmax(word) in fr_tokenizer.index_word else '' 
                                   for word in prediction[0]])

    return predicted_sentence

with gr.Blocks() as demo:
    name = gr.Textbox(lines=2, label="English Sentence")
    output = gr.Textbox(lines=2, label="French Sentence")
    title="English to French Translator"
    description="Enter an English sentence and click 'Translate' to get the French translation.",
    submit_btn = gr.Button("Translate")
    submit_btn.click(fn=translate_sentence, inputs=name, outputs=output, api_name="translate")


if __name__ == "__main__":
    demo.launch()
