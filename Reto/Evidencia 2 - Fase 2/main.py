import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
import nltk
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Global vectorizer instance
vectorizer = TfidfVectorizer()

# Initialize spaCy with pre-trained word vectors
nlp = spacy.load("en_core_web_md")

# Initialize the GUI environment
root = tk.Tk()
root.withdraw()  # we don't want a full GUI, so keep the root window from appearing

def select_file():
    return filedialog.askopenfilename()

def select_files():
    return filedialog.askopenfilenames()

def select_directory():
    return filedialog.askdirectory()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def read_and_preprocess(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return preprocess(content)

def calculate_similarity(doc1, doc2):
    docs = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(docs[0:1], docs[1:2])[0][0]

def calculate_semantic_similarity(doc1, doc2):
    doc1 = nlp(doc1)
    doc2 = nlp(doc2)
    return doc1.similarity(doc2)

def analyze_tense_and_voice(doc):
    tenses = []
    voices = []
    for token in doc:
        if token.tag_ in ["VBD", "VBN"]:  # Past tense verbs
            tenses.append("past")
        elif token.tag_ in ["VB", "VBP", "VBZ"]:  # Present tense verbs
            tenses.append("present")
        if "pass" in token.dep_:
            voices.append("passive")
        elif "subj" in token.dep_:
            voices.append("active")
    
    return tenses, voices

def compare_tense_and_voice(doc1, doc2):
    tenses1, voices1 = analyze_tense_and_voice(nlp(doc1))
    tenses2, voices2 = analyze_tense_and_voice(nlp(doc2))
    tense_change = set(tenses1) != set(tenses2)
    voice_change = set(voices1) != set(voices2)
    return tense_change, voice_change

def categorize_plagiarism(similarity):
    if similarity > 0.8:
        return 1  # 'High - Likely direct copy'
    elif similarity > 0.5:
        return 0.5  # 'Medium - Likely paraphrasing or summarizing'
    else:
        return 0  # 'Low'

def write_results_to_file(filepath, results):
    with open(filepath, 'w') as file:
        for result in results:
            file.write(f"{result}\n")

def compare_two_files(file1, file2):
    doc1 = read_and_preprocess(file1)
    doc2 = read_and_preprocess(file2)
    similarity_tfidf = calculate_similarity(doc1, doc2)
    similarity_semantic = calculate_semantic_similarity(doc1, doc2)
    tense_change, voice_change = compare_tense_and_voice(doc1, doc2)
    return similarity_tfidf, similarity_semantic, tense_change, voice_change

def compare_many_to_many(files1, files2):
    results = {}
    for file1 in files1:
        file1_results = []
        for file2 in files2:
            if file1 != file2:
                similarity_tfidf, similarity_semantic, tense_change, voice_change = compare_two_files(file1, file2)
                result_text = f"File1: {os.path.basename(file1)}, File2: {os.path.basename(file2)} - TF-IDF Similarity: {similarity_tfidf:.2f}, Semantic Similarity: {similarity_semantic:.2f}, Tense Change: {tense_change}, Voice Change: {voice_change}"
                file1_results.append(result_text)
        results[os.path.basename(file1)] = file1_results
    return results

def plot_similarity_scores(similarity_scores_tfidf, similarity_scores_semantic):
    plt.figure(figsize=(10, 5))
    plt.hist(similarity_scores_tfidf, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='TF-IDF')
    plt.hist(similarity_scores_semantic, bins=20, color='orange', edgecolor='black', alpha=0.7, label='Semantic')
    plt.xlabel('Similarity Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
def plot_roc_curve(y_true, y_pred):
    # Convert continuous scores to binary classification
    y_pred_binary = [1 if score > 0.8 else 0 for score in y_pred]  # Assuming 0.8 threshold for positive class
    fpr, tpr, _ = roc_curve(y_true, y_pred_binary, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    
def generate_summary_report(results):
    num_comparisons = len(results)
    if num_comparisons == 0:
        return "No comparisons were performed."

    average_tfidf = np.mean([float(re.search(r"TF-IDF Similarity: (\d+\.\d+)", res).group(1)) for res in results])
    average_semantic = np.mean([float(re.search(r"Semantic Similarity: (\d+\.\d+)", res).group(1)) for res in results])
    num_tense_changes = sum(["Tense Change: True" in res for res in results])
    num_voice_changes = sum(["Voice Change: True" in res for res in results])

    summary = (
        f"Summary of Comparisons:\n"
        f"Total Comparisons: {num_comparisons}\n"
        f"Average TF-IDF Similarity: {average_tfidf * 100:.2f}%\n"
        f"Average Semantic Similarity: {average_semantic * 100:.2f}%\n"
        f"Tense Changes Detected: {num_tense_changes} ({num_tense_changes/num_comparisons*100:.2f}% of cases)\n"
        f"Voice Changes Detected: {num_voice_changes} ({num_voice_changes/num_comparisons*100:.2f}% of cases)\n"
    )
    with open('resume.txt', 'w') as f:
        f.write(summary)
    return summary

def user_interface():
    print("Select the type of comparison:")
    print("1. Compare two files")
    print("2. Compare one file to multiple files")
    print("3. Compare multiple files to multiple files")
    print("4. Compare all files in a directory")
    choice = int(input("Enter your choice (1-4): "))
    results = []
    similarity_scores_tfidf = []
    similarity_scores_semantic = []

    if choice == 1:
        file1 = select_file()
        file2 = select_file()
        similarity_tfidf, similarity_semantic, tense_change, voice_change = compare_two_files(file1, file2)
        result_text = f"Files: {os.path.basename(file1)}, {os.path.basename(file2)} - TF-IDF Similarity: {similarity_tfidf:.2f}, Semantic Similarity: {similarity_semantic:.2f}, Tense Change: {tense_change}, Voice Change: {voice_change}"
        results.append(result_text)
        similarity_scores_tfidf.append(similarity_tfidf)
        similarity_scores_semantic.append(similarity_semantic)
    elif choice == 2:
        base_file = select_file()
        other_files = select_files()
        for file in other_files:
            similarity_tfidf, similarity_semantic, tense_change, voice_change = compare_two_files(base_file, file)
            result_text = f"Base File: {os.path.basename(base_file)}, Compared to: {os.path.basename(file)} - TF-IDF Similarity: {similarity_tfidf:.2f}, Semantic Similarity: {similarity_semantic:.2f}, Tense Change: {tense_change}, Voice Change: {voice_change}"
            results.append(result_text)
            similarity_scores_tfidf.append(similarity_tfidf)
            similarity_scores_semantic.append(similarity_semantic)
    elif choice == 3:
        files1 = select_files()
        files2 = select_files()
        for file1 in files1:
            for file2 in files2:
                if file1 != file2:
                    similarity_tfidf, similarity_semantic, tense_change, voice_change = compare_two_files(file1, file2)
                    result_text = f"File1: {os.path.basename(file1)}, File2: {os.path.basename(file2)} - TF-IDF Similarity: {similarity_tfidf:.2f}, Semantic Similarity: {similarity_semantic:.2f}, Tense Change: {tense_change}, Voice Change: {voice_change}"
                    results.append(result_text)
                    similarity_scores_tfidf.append(similarity_tfidf)
                    similarity_scores_semantic.append(similarity_semantic)
    elif choice == 4:
        directory = select_directory()
        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for file1 in files:
            for file2 in files:
                if file1 != file2:
                    similarity_tfidf, similarity_semantic, tense_change, voice_change = compare_two_files(file1, file2)
                    result_text = f"File1: {os.path.basename(file1)}, File2: {os.path.basename(file2)} - TF-IDF Similarity: {similarity_tfidf:.2f}, Semantic Similarity: {similarity_semantic:.2f}, Tense Change: {tense_change}, Voice Change: {voice_change}"
                    results.append(result_text)
                    similarity_scores_tfidf.append(similarity_tfidf)
                    similarity_scores_semantic.append(similarity_semantic)
    else:
        print("Invalid choice. Please select 1-4.")

    write_results_to_file('results.txt', results)
    
    generate_summary_report(results)
    
    if similarity_scores_tfidf and similarity_scores_semantic:
        plot_similarity_scores(similarity_scores_tfidf, similarity_scores_semantic)
        y_true = [categorize_plagiarism(score) for score in similarity_scores_tfidf]
        y_pred = [categorize_plagiarism(score) for score in similarity_scores_semantic]
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_pred)

if __name__ == '__main__':
    user_interface()