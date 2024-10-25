import spacy
import nltk

from NaiveRAG import NaiveRAG

nltk.download('wordnet')
from nltk.corpus import wordnet
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
class AdvancedRAG:
    def __init__(self):
        print('Hello in Naive RAG')

    def find_best_match(self,text_input, records):
        best_score = 0
        best_record = None
        for record in records:
            current_score = NaiveRAG().calculate_cosine_similarity(text_input, record)
            if current_score > best_score:
                best_score = current_score
                best_record = record
        return best_score, best_record

    def setup_vectorizer(self,records):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(records)
        return vectorizer, tfidf_matrix

    def find_best_match(self,query, vectorizer, tfidf_matrix):
        query_tfidf = vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, tfidf_matrix)
        best_index = similarities.argmax()  # Get the index of the highest similarity score
        best_score = similarities[0, best_index]
        return best_score, best_index


