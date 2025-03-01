import json
import spacy
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download("stopwords")

# Load NLP models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

class TopicCategory:
    def __init__(self, text: str):
        self.text = text

    def sentence_clustering(self):
        # Split text into sentences
        doc = nlp(self.text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        # Preprocessing function
        def preprocess(text):
            doc = nlp(text.lower())  # Convert to lowercase
            return [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords.words("english")]

        # Tokenized sentences for topic modeling
        tokenized_sentences = [preprocess(sent) for sent in sentences]

        # Create a dictionary and corpus for LDA
        dictionary = Dictionary(tokenized_sentences)
        corpus = [dictionary.doc2bow(sent) for sent in tokenized_sentences]

        # Train LDA Model to discover topics
        num_topics = 5  # Adjust based on dataset size
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)

        # Extract topic keywords
        topic_keywords = {i: [word for word, _ in lda.show_topic(i)] for i in range(num_topics)}

        # Sentence embeddings for clustering
        sentence_embeddings = np.array(model.encode(sentences))

        # Apply KMeans clustering to group sentences dynamically
        num_clusters = num_topics  # Use same number as LDA
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(sentence_embeddings)

        # Assign topics based on clusters
        def get_topic_label(cluster_id):
            return f"Topic {cluster_id + 1} ({', '.join(topic_keywords[cluster_id][:3])})"

        sentence_data = [
            {
                "text": sent,  
                "label": str(clusters[i]),  # Numeric cluster ID
                "label_name": get_topic_label(clusters[i])  # Topic name
            }
            for i, sent in enumerate(sentences)
        ]

        return sentence_data

if __name__ == "__main__":
    topic = TopicCategory(r"C:\Users\dell\Entrepreneurship\Engineering\ai_repo\NLP\Epidemology\TopicCategory\text.txt")
    topic.sentence_clustering()