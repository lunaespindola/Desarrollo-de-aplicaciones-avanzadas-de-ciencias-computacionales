"""
    author: @LunaEspindola
    date: 2024/04/03
    Description:
        This code calculates the cosine similarity between vectors.
"""

# Importing libraries
import numpy as np  
from sklearn.metrics.pairwise import cosine_similarity


# Function to strip the text
def strip_text(text):
    return text.lower().strip()

# Function to eliminate the repeated words
def eliminate_repeated_words(text):
    return ' '.join(set(text.split()))

# Function to calculate the cosine similarity
def cosine_similarity(text1, text2):
    # Stripping the text
    text1 = strip_text(text1)
    text2 = strip_text(text2)
    
    # Eliminating repeated words
    text1 = eliminate_repeated_words(text1)
    text2 = eliminate_repeated_words(text2)
    
    # Tokenizing the text
    text1 = text1.split()
    text2 = text2.split()
    
    # Creating the dictionary of words
    words = set(text1).union(set(text2))
    
    # Creating the vectors
    vector1 = [text1.count(word) for word in words]
    vector2 = [text2.count(word) for word in words]
    
    # Calculating the cosine similarity
    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    
    return cosine_similarity

def main():
    text1 = "I am a student"
    text2 = "I am a teacher"
    
    print(f"The cosine similarity between the texts is: {cosine_similarity(text1, text2)}")
    
main()
