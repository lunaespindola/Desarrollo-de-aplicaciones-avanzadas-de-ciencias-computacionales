import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog
import os   

# Function to find exact matching phrases
def find_exact_matches(text1, text2, n=5):
    # Tokenize both texts
    tokens1 = word_tokenize(text1.lower())
    tokens2 = word_tokenize(text2.lower())
    
    # Generate n-grams (sequences of n words) for both texts
    ngrams1 = set([' '.join(tokens1[i:i+n]) for i in range(len(tokens1)-n+1)])
    ngrams2 = set([' '.join(tokens2[i:i+n]) for i in range(len(tokens2)-n+1)])
    
    # Find common n-grams in both texts
    matches = ngrams1.intersection(ngrams2)
    
    return matches

# Function to calculate cosine similarity
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the texts
    tfidf = vectorizer.fit_transform([text1, text2])
    
    # Compute cosine similarity between the first and second document
    sim_score = cosine_similarity(tfidf[0:1], tfidf[1:2])
    
    return sim_score[0][0]

# Example usage
text1 = filedialog.askopenfilename()
text2 = filedialog.askopenfilename()

# Read the text files
with open(text1, 'r', encoding='utf-8') as file:
    text1 = file.read()
    
with open(text2, 'r', encoding='utf-8') as file:
    text2 = file.read()
    
    
#check to multiple files


# Find exact matches
matches = find_exact_matches(text1, text2, n=5)
print("Exact Matches:", matches)

# Calculate cosine similarity
similarity_score = calculate_cosine_similarity(text1, text2)
similarity_score *= 100
print("Cosine Similarity Score:", similarity_score)
