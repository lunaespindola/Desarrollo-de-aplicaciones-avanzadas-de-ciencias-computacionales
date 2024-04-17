# Definicion de parrafos
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

# Function to eliminate punctuation
def eliminate_punctuation(text):
    punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    text = ''.join([char for char in text if char not in punctuation])
    return text

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

# Change the result of the cosine similarity to a percentage
def cosine_similarity_percentage(text1, text2):
    return cosine_similarity(text1, text2) * 100


def main():
    text1 = "La filosofía nació en el momento en que se descubrió que existe una diferencia esencial entre apariencia y realidad. "
    text2 = "Las cosas no son en realidad tal y como se nos aparecen: la superficie de una mesa parece sólida e inmóvil, sin embargo, según la física es altamente porosa y está cargada de partículas eléctricas."
    text3 = "Se ha dicho que la filosofía comienza con el asombro."
    text4 = "Puede decirse también que comienza con la curiosidad."
    text5 = "Filosofar es intentar abrir la puerta que nos permite cruzar el umbral de la apariencia y entrar en el mundo de la realidad."
    text6 = "Además, la filosofía exige valor, porque no sabemos qué es lo que hay al otro lado de la puerta hasta que la abrimos."

    text7 = "La filosofía existe desde el momento en que se entendio la existencia de una diferencia esencial entre apariencia y realidad."
    text8 = "Las cosas no son realmente lo que parecen: la superficie de la mesa parece sólida e inmóvil, pero según la física es muy porosa y está llena de partículas eléctricas."
    text9 = "Se ha dicho que la filosofía comienza con un milagro."
    text10 = "También se podría decir que comienza con curiosidad."
    text11 = "El filósofo intenta abrir la puerta que nos permite cruzar el umbral de la apariencia y entrar en el mundo de la realidad."
    text12 = "Además, la filosofía requiere coraje, porque no sabemos qué hay al otro lado de la puerta hasta que la abrimos."
    
    paragraph1 = "La filosofía nació en el momento en que se descubrió que existe una diferencia esencial entre apariencia y realidad. Las cosas no son en realidad tal y como se nos aparecen: la superficie de una mesa parece sólida e inmóvil, sin embargo, según la física es altamente porosa y está cargada de partículas eléctricas. Se ha dicho que la filosofía comienza con el asombro. Puede decirse también que comienza con la curiosidad. Filosofar es intentar abrir la puerta que nos permite cruzar el umbral de la apariencia y entrar en el mundo de la realidad. Además, la filosofía exige valor, porque no sabemos qué es lo que hay al otro lado de la puerta hasta que la abrimos. "
    paragraph2 = "La filosofía existe desde el momento en que se entendio la existencia de una diferencia esencial entre apariencia y realidad. Las cosas no son realmente lo que parecen: la superficie de la mesa parece sólida e inmóvil, pero según la física es muy porosa y está llena de partículas eléctricas. Se ha dicho que la filosofía comienza con un milagro. También se podría decir que comienza con curiosidad. El filósofo intenta abrir la puerta que nos permite cruzar el umbral de la apariencia y entrar en el mundo de la realidad. Además, la filosofía requiere coraje, porque no sabemos qué hay al otro lado de la puerta hasta que la abrimos."
    
    # Comparing paragraph 1 with itself
    print(f"The cosine similarity between the text 1 and 2 is: {cosine_similarity_percentage(text1, text2)}%")
    print(f"The cosine similarity between the text 1 and 3 is: {cosine_similarity_percentage(text1, text3)}%")
    print(f"The cosine similarity between the text 1 and 4 is: {cosine_similarity_percentage(text1, text4)}%")
    print(f"The cosine similarity between the text 1 and 5 is: {cosine_similarity_percentage(text1, text5)}%")
    print(f"The cosine similarity between the text 1 and 6 is: {cosine_similarity_percentage(text1, text6)}%")
    
    print(f"The cosine similarity between the text 2 and 1 is: {cosine_similarity_percentage(text2, text1)}%")
    print(f"The cosine similarity between the text 2 and 3 is: {cosine_similarity_percentage(text2, text3)}%")
    print(f"The cosine similarity between the text 2 and 4 is: {cosine_similarity_percentage(text2, text4)}%")
    print(f"The cosine similarity between the text 2 and 5 is: {cosine_similarity_percentage(text2, text5)}%")
    print(f"The cosine similarity between the text 2 and 6 is: {cosine_similarity_percentage(text2, text6)}%")
    
    print(f"The cosine similarity between the text 3 and 1 is: {cosine_similarity_percentage(text3, text1)}%")
    print(f"The cosine similarity between the text 3 and 2 is: {cosine_similarity_percentage(text3, text2)}%")
    print(f"The cosine similarity between the text 3 and 4 is: {cosine_similarity_percentage(text3, text4)}%")
    print(f"The cosine similarity between the text 3 and 5 is: {cosine_similarity_percentage(text3, text5)}%")
    print(f"The cosine similarity between the text 3 and 6 is: {cosine_similarity_percentage(text3, text6)}%")
    
    print(f"The cosine similarity between the text 4 and 1 is: {cosine_similarity_percentage(text4, text1)}%")
    print(f"The cosine similarity between the text 4 and 2 is: {cosine_similarity_percentage(text4, text2)}%")
    print(f"The cosine similarity between the text 4 and 3 is: {cosine_similarity_percentage(text4, text3)}%")
    print(f"The cosine similarity between the text 4 and 5 is: {cosine_similarity_percentage(text4, text5)}%")
    print(f"The cosine similarity between the text 4 and 6 is: {cosine_similarity_percentage(text4, text6)}%")
    
    print(f"The cosine similarity between the text 5 and 1 is: {cosine_similarity_percentage(text5, text1)}%")
    print(f"The cosine similarity between the text 5 and 2 is: {cosine_similarity_percentage(text5, text2)}%")
    print(f"The cosine similarity between the text 5 and 3 is: {cosine_similarity_percentage(text5, text3)}%")
    print(f"The cosine similarity between the text 5 and 4 is: {cosine_similarity_percentage(text5, text4)}%")
    print(f"The cosine similarity between the text 5 and 6 is: {cosine_similarity_percentage(text5, text6)}%")
    
    print(f"The cosine similarity between the text 6 and 1 is: {cosine_similarity_percentage(text6, text1)}%")
    print(f"The cosine similarity between the text 6 and 2 is: {cosine_similarity_percentage(text6, text2)}%")
    print(f"The cosine similarity between the text 6 and 3 is: {cosine_similarity_percentage(text6, text3)}%")
    print(f"The cosine similarity between the text 6 and 4 is: {cosine_similarity_percentage(text6, text4)}%")
    print(f"The cosine similarity between the text 6 and 5 is: {cosine_similarity_percentage(text6, text5)}%")
    
    
    # Comparing paragraph 2 with itself
    print(f"The cosine similarity between the text 7 and 8 is: {cosine_similarity_percentage(text7, text8)}%")
    print(f"The cosine similarity between the text 7 and 9 is: {cosine_similarity_percentage(text7, text9)}%")
    print(f"The cosine similarity between the text 7 and 10 is: {cosine_similarity_percentage(text7, text10)}%")
    print(f"The cosine similarity between the text 7 and 11 is: {cosine_similarity_percentage(text7, text11)}%")
    print(f"The cosine similarity between the text 7 and 12 is: {cosine_similarity_percentage(text7, text12)}%")
    
    print(f"The cosine similarity between the text 8 and 7 is: {cosine_similarity_percentage(text8, text7)}%")
    print(f"The cosine similarity between the text 8 and 9 is: {cosine_similarity_percentage(text8, text9)}%")
    print(f"The cosine similarity between the text 8 and 10 is: {cosine_similarity_percentage(text8, text10)}%")
    print(f"The cosine similarity between the text 8 and 11 is: {cosine_similarity_percentage(text8, text11)}%")
    print(f"The cosine similarity between the text 8 and 12 is: {cosine_similarity_percentage(text8, text12)}%")
    
    print(f"The cosine similarity between the text 9 and 7 is: {cosine_similarity_percentage(text9, text7)}%")
    print(f"The cosine similarity between the text 9 and 8 is: {cosine_similarity_percentage(text9, text8)}%")
    print(f"The cosine similarity between the text 9 and 10 is: {cosine_similarity_percentage(text9, text10)}%")
    print(f"The cosine similarity between the text 9 and 11 is: {cosine_similarity_percentage(text9, text11)}%")
    print(f"The cosine similarity between the text 9 and 12 is: {cosine_similarity_percentage(text9, text12)}%")
    
    print(f"The cosine similarity between the text 10 and 7 is: {cosine_similarity_percentage(text10, text7)}%")
    print(f"The cosine similarity between the text 10 and 8 is: {cosine_similarity_percentage(text10, text8)}%")
    print(f"The cosine similarity between the text 10 and 9 is: {cosine_similarity_percentage(text10, text9)}%")
    print(f"The cosine similarity between the text 10 and 11 is: {cosine_similarity_percentage(text10, text11)}%")
    print(f"The cosine similarity between the text 10 and 12 is: {cosine_similarity_percentage(text10, text12)}%")
    
    print(f"The cosine similarity between the text 11 and 7 is: {cosine_similarity_percentage(text11, text7)}%")
    print(f"The cosine similarity between the text 11 and 8 is: {cosine_similarity_percentage(text11, text8)}%")
    print(f"The cosine similarity between the text 11 and 9 is: {cosine_similarity_percentage(text11, text9)}%")
    print(f"The cosine similarity between the text 11 and 10 is: {cosine_similarity_percentage(text11, text10)}%")
    print(f"The cosine similarity between the text 11 and 12 is: {cosine_similarity_percentage(text11, text12)}%")
    
    print(f"The cosine similarity between the text 12 and 7 is: {cosine_similarity_percentage(text12, text7)}%")
    print(f"The cosine similarity between the text 12 and 8 is: {cosine_similarity_percentage(text12, text8)}%")
    print(f"The cosine similarity between the text 12 and 9 is: {cosine_similarity_percentage(text12, text9)}%")
    print(f"The cosine similarity between the text 12 and 10 is: {cosine_similarity_percentage(text12, text10)}%")
    print(f"The cosine similarity between the text 12 and 11 is: {cosine_similarity_percentage(text12, text11)}%")
    
    # Comparing paragraph 1 with paragraph 2
    print(f"The cosine similarity between the paragraph 1 and 2 is: {cosine_similarity_percentage(paragraph1, paragraph2)}%")
    
    
main()