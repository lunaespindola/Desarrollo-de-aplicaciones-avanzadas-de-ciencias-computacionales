'''
    @Author: @Lunaespindola
    @Date: 2024/04/16
    @Description: This is the code to check plagiarism in a text file or multiple text files and returns a brief on a txt.
'''

# Importing the necessary libraries
import os
import sys
import re
import string
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Downloading the necessary resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Function to read the text files
def read_files(file_path):
    '''
    Function to read the text files

    Args:
        file_path (str): The path of the file to be read

    Returns:
        str: The content of the file
    '''
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# Function to clean the text
def clean_text(text):
    '''
    Function to clean the text

    Args:
        text (str): The text to be cleaned

    Returns:
        str: The cleaned text
    '''
    # Removing the special characters
    text = re.sub(r'\W', ' ', text)
    # Removing the digits
    text = re.sub(r'\d', ' ', text)
    # Removing the single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Removing the single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Converting to Lowercase
    text = text.lower()
    return text

# Function to remove the stopwords
def remove_stopwords(text):
    '''
    Function to remove the stopwords

    Args:
        text (str): The text from which the stopwords are to be removed

    Returns:
        str: The text with the stopwords removed
    '''
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to stem the text
def stem_text(text):
    '''
    Function to stem the text

    Args:
        text (str): The text to be stemmed

    Returns:
        str: The stemmed text
    '''
    ps = PorterStemmer()
    words = word_tokenize(text)
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Function to lemmatize the text
def lemmatize_text(text):
    '''
    Function to lemmatize the text

    Args:
        text (str): The text to be lemmatized

    Returns:
        str: The lemmatized text
    '''
    wnl = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [wnl.lemmatize(word) for word in words]
    return ' '.join(words)

# Function to check plagiarism
def check_plagiarism(file1, file2, method='cosine'):
    '''
    Function to check plagiarism

    Args:
        file1 (str): The path of the first file
        file2 (str): The path of the second file
        method (str): The method to be used for checking plagiarism, default is 'cosine'

    Returns:
        float: The similarity score
    '''
    # Reading the files
    content1 = read_files(file1)
    content2 = read_files(file2)

    # Cleaning the text
    content1 = clean_text(content1)
    content2 = clean_text(content2)

    # Removing the stopwords
    content1 = remove_stopwords(content1)
    content2 = remove_stopwords(content2)

    # Stemming the text
    content1 = stem_text(content1)
    content2 = stem_text(content2)

    # Lemmatizing the text
    content1 = lemmatize_text(content1)
    content2 = lemmatize_text(content2)

    # Creating the CountVectorizer object
    cv = CountVectorizer()

    # Fitting the data
    count_matrix = cv.fit_transform([content1, content2])

    # Computing the similarity score
    if method == 'cosine':
        similarity = sklearn.metrics.pairwise.cosine_similarity(count_matrix)
    else:
        similarity = sklearn.metrics.pairwise.euclidean_distances(count_matrix)
    
    return similarity[0][1]

# Function to plot the similarity score
def plot_similarity_score(similarity_scores):
    '''
    Function to plot the similarity score

    Args:
        similarity_scores (list): The list of similarity scores
    '''
    plt.plot(similarity_scores)
    plt.xlabel('File Pair')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Score of File Pairs')
    plt.xticks(ticks=range(len(similarity_scores)))
    plt.grid()
    plt.tight_layout()
    plt.savefig('similarity_score.png')
    plt.show()
    
# Function to check plagiarism in a directory
def check_plagiarism_directory(directory, method='cosine'):
    '''
    Function to check plagiarism in a directory

    Args:
        directory (str): The path of the directory
        method (str): The method to be used for checking plagiarism, default is 'cosine'
    '''
    files = os.listdir(directory)
    similarity_scores = []
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            file1 = os.path.join(directory, files[i])
            file2 = os.path.join(directory, files[j])
            similarity = check_plagiarism(file1, file2, method)
            similarity_scores.append(similarity)
    plot_similarity_score(similarity_scores)
    
# Functipn to check plagiarism from one file to multiple files
def check_plagiarism_file_to_files(file, files, method='cosine'):
    '''
    Function to check plagiarism from one file to multiple files

    Args:
        file (str): The path of the file
        files (list): The list of paths of the files
        method (str): The method to be used for checking plagiarism, default is 'cosine'
    '''
    similarity_scores = []
    for f in files:
        similarity = check_plagiarism(file, f, method)
        similarity_scores.append(similarity)
    plot_similarity_score(similarity_scores)
    
# Function to check plagiarism from multiple files to multiple files
def check_plagiarism_files_to_files(files1, files2, method='cosine'):
    '''
    Function to check plagiarism from multiple files to multiple files

    Args:
        files1 (list): The list of paths of the first set of files
        files2 (list): The list of paths of the second set of files
        method (str): The method to be used for checking plagiarism, default is 'cosine'
    '''
    similarity_scores = []
    for f1 in files1:
        for f2 in files2:
            similarity = check_plagiarism(f1, f2, method)
            similarity_scores.append(similarity)
    plot_similarity_score(similarity_scores)
    
# Function to check plagiarism in a directory and write the results to a file
def check_plagiarism_directory_to_file(directory, method='cosine'):
    '''
    Function to check plagiarism in a directory and write the results to a file

    Args:
        directory (str): The path of the directory
        method (str): The method to be used for checking plagiarism, default is 'cosine'
    '''
    files = os.listdir(directory)
    similarity_scores = []
    with open('results.txt', 'w') as file:
        for i in range(len(files)):
            for j in range(i+1, len(files)):
                file1 = os.path.join(directory, files[i])
                file2 = os.path.join(directory, files[j])
                similarity = check_plagiarism(file1, file2, method)
                similarity_scores.append(similarity)
                file.write(f'The similarity score between {files[i]} and {files[j]} is {similarity}\n')
    plot_similarity_score(similarity_scores)
    
# Main function
def main():
    '''
    Main function
    '''
    # # Checking the number of arguments
    # if len(sys.argv) < 2:
    #     print('Usage: python main.py <file_path>')
    #     sys.exit(1)
    
    # # Checking if the file exists
    # file_path = sys.argv[1]
    # if not os.path.exists(file_path):
    #     print('The file does not exist')
    #     sys.exit(1)
    
    directory = 'Resumen Texto Otros'
        
    file_path_1 = 'Resumen Texto Otros/FID-01.txt'
    file_path_2 = 'Resumen Texto Otros/FID-02.txt'
    file_path_3 = 'Resumen Texto Otros/FID-03.txt'
    file_path_4 = 'Resumen Texto Otros/FID-04.txt'
    file_path_5 = 'Resumen Texto Otros/FID-05.txt'
    file_path_6 = 'Resumen Texto Otros/FID-06.txt'
    file_path_7 = 'Resumen Texto Otros/FID-07.txt'
    file_path_8 = 'Resumen Texto Otros/FID-08.txt'
    file_path_9 = 'Resumen Texto Otros/FID-09.txt'
    file_path_10 = 'Resumen Texto Otros/FID-10.txt'
    
    # Get file names
    name1 = os.path.basename(file_path_1)
    name2 = os.path.basename(file_path_2)
    name3 = os.path.basename(file_path_3)
    name4 = os.path.basename(file_path_4)
    name5 = os.path.basename(file_path_5)
    name6 = os.path.basename(file_path_6)
    name7 = os.path.basename(file_path_7)
    name8 = os.path.basename(file_path_8)
    name9 = os.path.basename(file_path_9)
    name10 = os.path.basename(file_path_10)
    
    # Reading the file
    file1 = read_files(file_path_1)
    file2 = read_files(file_path_2)
    
    # Checking plagiarism
    similarity = check_plagiarism(file_path_1, file_path_2)
    print(f'The similarity score between {name1} and {name2} is {similarity}')
    
    # Checking plagiarism from one file to multiple files
    check_plagiarism_file_to_files(file_path_1, [file_path_2, file_path_3, file_path_4, file_path_5, file_path_6, file_path_7, file_path_8, file_path_9, file_path_10])
    
    # Checking plagiarism in a directory
    check_plagiarism_directory(directory)
    
    # Checking plagiarism in a directory and writing the results to a file
    check_plagiarism_directory_to_file(directory)
    

# Checking if the script is run as the main script
if __name__ == '__main__':
    main()