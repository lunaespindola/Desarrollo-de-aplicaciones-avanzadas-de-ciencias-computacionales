import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import tkinter as tk
from tkinter import filedialog
import re


nltk.download('punkt')
nltk.download('stopwords')

# Global vectorizer instance
vectorizer = TfidfVectorizer()


# Initialize the GUI environment
root = tk.Tk()
root.withdraw()  # we don't want a full GUI, so keep the root window from appearing

# Function to select a single file
def select_file():
    return filedialog.askopenfilename()

# Function to select multiple files
def select_files():
    return filedialog.askopenfilenames()

# Function to select a directory
def select_directory():
    return filedialog.askdirectory()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)  # Return a single string

def read_and_preprocess(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return preprocess(content)

def calculate_similarity(doc1, doc2):
    docs = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(docs[0:1], docs[1:2])[0][0]

def categorize_plagiarism(similarity):
    if similarity > 0.8:
        return 'High - Might be whole phrases copied or only minor rewording'
    elif similarity > 0.5:
        return 'Medium - Could be substantial rewording or summarizing'
    else:
        return 'Low - Likely reordering or distant summarizing'

def write_results_to_file(filepath, results):
    with open(filepath, 'w') as file:
        for result in results:
            file.write(f"{result}\n")

# Compare two files
def compare_two_files(file1, file2):
    doc1 = read_and_preprocess(file1)
    doc2 = read_and_preprocess(file2)
    similarity = calculate_similarity(doc1, doc2)
    return similarity

# Compare one file to multiple files
def compare_one_to_many(base_file, other_files):
    base_doc = read_and_preprocess(base_file)
    results = {}
    for file in other_files:
        other_doc = read_and_preprocess(file)
        similarity = calculate_similarity(base_doc, other_doc)
        results[file] = similarity
    return results

# Compare multiple files to multiple files
def compare_many_to_many(files1, files2):
    results = {}
    for file1 in files1:
        results[file1] = compare_one_to_many(file1, files2)
    return results

# Compare all files in a directory
def compare_directory(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return compare_many_to_many(files, files)

def plot_similarity_scores(similarity_scores):
    plt.hist(similarity_scores, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Similarity Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Updated user interface for selecting comparison type using file dialogs
import os  # Make sure you have this import if it's not already included

def user_interface():
    print("Select the type of comparison:")
    print("1. Compare two files")
    print("2. Compare one file to multiple files")
    print("3. Compare multiple files to multiple files")
    print("4. Compare all files in a directory")
    choice = int(input("Enter your choice (1-4): "))
    results = []

    if choice == 1:
        file1 = select_file()
        file2 = select_file()
        similarity = compare_two_files(file1, file2)
        category = categorize_plagiarism(similarity)
        similarity *= 100  # Multiply by 100 to get a percentage
        result_text = f"Files: {os.path.basename(file1)}, {os.path.basename(file2)} - Similarity: {similarity:.2f}%, Category: {category}"
        results.append(result_text)
    elif choice == 2:
        base_file = select_file()
        other_files = select_files()
        similarities = compare_one_to_many(base_file, other_files)
        for file, similarity in similarities.items():
            similarity *= 100  # Multiply by 100 to get a percentage
            category = categorize_plagiarism(similarity)
            result_text = f"Base File: {os.path.basename(base_file)}, Compared to: {os.path.basename(file)} - Similarity: {similarity:.2f}%, Category: {category}"
            results.append(result_text)
    elif choice == 3:
        files1 = select_files()
        files2 = select_files()
        all_similarities = compare_many_to_many(files1, files2)
        for file1, similarities in all_similarities.items():
            for file2, similarity in similarities.items():
                similarity *= 100  # Multiply by 100 to get a percentage
                category = categorize_plagiarism(similarity)
                result_text = f"File1: {os.path.basename(file1)}, File2: {os.path.basename(file2)} - Similarity: {similarity:.2f}%, Category: {category}"
                results.append(result_text)
    elif choice == 4:
        directory = select_directory()
        all_similarities = compare_directory(directory)
        for file1, similarities in all_similarities.items():
            for file2, similarity in similarities.items():
                if file1 != file2:  # Prevent comparing the file with itself
                    similarity *= 100  # Multiply by 100 to get a percentage
                    category = categorize_plagiarism(similarity)
                    result_text = f"File1: {os.path.basename(file1)}, File2: {os.path.basename(file2)} - Similarity: {similarity:.2f}%, Category: {category}"
                    results.append(result_text)
    else:
        print("Invalid choice. Please select 1-4.")

    # Writing results to the file after all operations
    write_results_to_file('results.txt', results)

    # Calculate similarity scores for evaluation
    similarity_scores = [float(re.search(r'Similarity: (\d+\.\d+)%', result).group(1)) for result in results]

    # Plot similarity scores, ROC curve, and confusion matrix
    plot_similarity_scores(similarity_scores)
    plot_roc_curve(np.ones(len(similarity_scores)), similarity_scores)
    plot_confusion_matrix(np.ones(len(similarity_scores)), np.ones(len(similarity_scores)), labels=[0, 1])  # Assuming all documents are plagiarized

if __name__ == '__main__':
    user_interface()
