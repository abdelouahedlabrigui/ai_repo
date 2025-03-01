import json
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class Preprocessing:
    def __init__(self, filename):
        self.filename = filename

    def preprocess_text(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt_tab')

        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()


        with open(f'{self.filename}', 'r', encoding='utf-8') as file:
            text = file.read()

        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        text = text.lower()

        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        filtered_words = [word for word in words if word not in stop_words]

        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

        preprocessed_data = {
            'original_text': text,
            'sentences': sentences,
            'words': words,
            'filtered_words': filtered_words,
            'stemmed_words': stemmed_words,
            'lemmatized_words': lemmatized_words
        }

        return preprocessed_data
    

def main():
    filename = r'C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\The SPARS Pandemic 2025-2028.txt'
    preprocessor = Preprocessing(filename)
    preprocessed_data = json.dumps(preprocessor.preprocess_text(), indent=4)
    written_file = r'C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Preprocessed.json'
    with open(f'{written_file}', 'w', encoding='utf-8') as file:
        file.write(preprocessed_data)
   

if __name__ == '__main__':
    main()