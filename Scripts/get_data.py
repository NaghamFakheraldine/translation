import pandas as pd
import os
import shutil

def get_data():
    repo_path = 'zaka-aic-2024'
    data_path = os.path.join(repo_path, 'Data')

    if not os.path.exists(repo_path) or not os.listdir(repo_path):
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path) 
        os.system('git clone https://github.com/NaghamFakheraldine/zaka-aic-2024.git')

    english_csv = os.path.join(data_path, 'en.csv')
    french_csv = os.path.join(data_path, 'fr.csv')
    
    if not os.path.exists(english_csv) or not os.path.exists(french_csv):
        raise FileNotFoundError("The required CSV files are missing from the repository.")
    
    english_sentences = pd.read_csv(english_csv, header=None)
    french_sentences = pd.read_csv(french_csv, header=None)

    if english_sentences.empty or french_sentences.empty:
        raise ValueError("One or both of the CSV files are empty.")

    df = pd.concat([english_sentences, french_sentences], axis=1)
    df.columns = ["English", "French"]
    
    if not os.path.exists('../data'):
        os.makedirs('data')

    df.to_csv("../data/translation_data.csv", index=False)

if __name__ == "__main__":
    get_data()
