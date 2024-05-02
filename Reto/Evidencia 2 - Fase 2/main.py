'''
    @author: @Lunaespindola04
    @date: 2024/05/01
    @description: This script is a plagiarism detection tool that compares two text files and determines the similarity between them.
'''

# Import libraries
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
import pypandoc
import sys
import subprocess

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Global vectorizer instance
vectorizer = TfidfVectorizer()

# Initialize spaCy with pre-trained word vectors
nlp = spacy.load("en_core_web_md")

# Initialize the GUI environment
root = tk.Tk()
root.withdraw()  

#----------------- File selection functions -----------------#
def select_file():
    return filedialog.askopenfilename()

def select_files():
    return filedialog.askopenfilenames()

def select_directory():
    return filedialog.askdirectory()

#----------------- Text preprocessing functions -----------------#
def preprocess(text):
    '''
        Preprocesses the input text by tokenizing, removing stopwords, and stemming the words.
        @args: text (str): The input text to be preprocessed
        @returns: str: The preprocessed text
    '''
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def read_and_preprocess(filepath):
    '''
        Reads the content of a file and preprocesses it.
        @args: filepath (str): The path to the file to be read
        @returns: str: The preprocessed text content of the file
    '''
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return preprocess(content)

#----------------- Text comparison functions -----------------#
def calculate_similarity(doc1, doc2):
    '''
        Calculates the cosine similarity between two documents using TF-IDF vectors.
        @args: doc1 (str): The first document
               doc2 (str): The second document
        @returns: float: The cosine similarity between the two documents
    '''
    docs = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(docs[0:1], docs[1:2])[0][0]

def calculate_semantic_similarity(doc1, doc2):
    '''
        Calculates the semantic similarity between two documents using spaCy's pre-trained word vectors.
        @args: doc1 (str): The first document
               doc2 (str): The second document
        @returns: float: The semantic similarity between the two documents
    '''
    doc1 = nlp(doc1)
    doc2 = nlp(doc2)
    return doc1.similarity(doc2)

#----------------- Plagiarism detection functions -----------------#
def analyze_tense_and_voice(doc):
    '''
        Analyzes the tense and voice of verbs in a given document.
        @args: doc (spacy.Doc): The input document
        @returns: list: A list of tenses found in the document  
    '''
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
    '''
        Compares the tense and voice of two documents.
        @args: doc1 (str): The first document
               doc2 (str): The second document
        @returns: tuple: A tuple of booleans indicating whether there is a tense change and a voice change between the two documents
    '''
    tenses1, voices1 = analyze_tense_and_voice(nlp(doc1))
    tenses2, voices2 = analyze_tense_and_voice(nlp(doc2))
    tense_change = set(tenses1) != set(tenses2)
    voice_change = set(voices1) != set(voices2)
    return tense_change, voice_change

def categorize_plagiarism(similarity):
    '''
        Categorizes the similarity score into different levels of plagiarism.
        @args: similarity (float): The similarity score between two documents
        @returns: int: The level of plagiarism detected (0: Low, 0.5: Medium, 1: High)
    '''
    if similarity > 0.8:
        return 1  # 'High - Likely direct copy'
    elif similarity > 0.5:
        return 0.5  # 'Medium - Likely paraphrasing or summarizing'
    else:
        return 0  # 'Low'

#----------------- File I/O and reporting function -----------------#
def write_results_to_file(filepath, results):
    '''
        Writes the comparison results to a text file.
        @args: filepath (str): The path to the output file
                results (list): A list of comparison results
        @returns: None 
    '''
    with open(filepath, 'w') as file:
        for result in results:
            file.write(f"{result}\n")
            
def generate_summary_report(results):
    '''
        Generates a summary report of the comparison results.
        @args: results (list): A list of comparison results
        @returns: str: A summary report of the comparison results
    '''
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
    with open('./Results/resume.txt', 'w') as f:
        f.write(summary)
    return summary

def create_dataframe(results):
    '''
        Creates a pandas DataFrame from the comparison results.
        @args: results (list): A list of comparison results
        @returns: pd.DataFrame: A DataFrame containing the comparison results
    '''
    data = []
    for result in results:
        match_file1 = re.search(r"File1: ([^,]+),", result)
        match_file2 = re.search(r"File2: ([^,]+) -", result)
        match_tfidf = re.search(r"TF-IDF Similarity: (\d+\.\d+)", result)
        match_semantic = re.search(r"Semantic Similarity: (\d+\.\d+)", result)
        match_tense = re.search(r"Tense Change: (.+),", result)
        match_voice = re.search(r"Voice Change: (.+)", result)

        if all([match_file1, match_file2, match_tfidf, match_semantic, match_tense, match_voice]):
            file1 = match_file1.group(1)
            file2 = match_file2.group(1)
            tfidf_similarity = float(match_tfidf.group(1))
            semantic_similarity = float(match_semantic.group(1))
            tense_change = match_tense.group(1)
            voice_change = match_voice.group(1)
            data.append([file1, file2, tfidf_similarity, semantic_similarity, tense_change, voice_change])
        else:
            # Handle cases where some information might be missing
            print(f"Could not parse one or more fields in the result: {result}")
    return pd.DataFrame(data, columns=['File1', 'File2', 'TF-IDF Similarity', 'Semantic Similarity', 'Tense Change', 'Voice Change'])

def save_results_to_csv(results, filepath):
    '''
        Saves the comparison results to a CSV file.
        @args: results (list): A list of comparison results
                filepath (str): The path to the output CSV file
        @returns: None
    '''
    df = create_dataframe(results)
    df.to_csv(filepath, index=False)
    
def create_md_table(results):
    '''
        Creates a markdown table from the comparison results.
        @args: results (list): A list of comparison results
        @returns: str: A markdown table containing the comparison results
    '''
    table = "| File1 | File2 | TF-IDF Similarity | Semantic Similarity | Tense Change | Voice Change |\n"
    table += "|-------|-------|-------------------|---------------------|--------------|-------------|\n"
    for result in results:
        # Extract fields using regex with error handling
        match_file1 = re.search(r"File1: ([^,]+),", result)
        match_file2 = re.search(r"File2: ([^,]+) -", result)
        match_tfidf = re.search(r"TF-IDF Similarity: (\d+\.\d+)", result)
        match_semantic = re.search(r"Semantic Similarity: (\d+\.\d+)", result)
        match_tense = re.search(r"Tense Change: (True|False),", result)
        match_voice = re.search(r"Voice Change: (True|False)", result)

        if not all([match_file1, match_file2, match_tfidf, match_semantic, match_tense, match_voice]):
            print(f"Could not parse one or more fields in the result: {result}")
            continue

        file1 = match_file1.group(1).strip()
        file2 = match_file2.group(1).strip()
        tfidf_similarity = float(match_tfidf.group(1))
        semantic_similarity = float(match_semantic.group(1))
        tense_change = match_tense.group(1)
        voice_change = match_voice.group(1)

        # Append data to the table
        table += f"| {file1} | {file2} | {tfidf_similarity:.2f} | {semantic_similarity:.2f} | {tense_change} | {voice_change} |\n"
    
    return table


def save_results_to_md(results, filepath):
    '''
        Saves the comparison results to a markdown file.
        @args: results (list): A list of comparison results
                filepath (str): The path to the output markdown file
        @returns: None
    '''
    table = create_md_table(results)
    with open(filepath, 'w') as file:
        file.write(table)
        
def convert_to_pdf(input_file, output_file):
    '''
        Converts a markdown file to a PDF file using pandoc.
        @args: input_file (str): The path to the input markdown file
                output_file (str): The path to the output PDF file
        @returns: None
    '''
    pypandoc.convert_file(input_file, 'pdf', outputfile=output_file)
    
def convert_to_html(input_file, output_file):
    '''
        Converts a markdown file to an HTML file using pandoc.
        @args: input_file (str): The path to the input markdown file
                output_file (str): The path to the output HTML file
        @returns: None
    '''
    pypandoc.convert_file(input_file, 'html', outputfile=output_file)

#----------------- File comparing functions -----------------#
def compare_two_files(file1, file2):
    '''
        Compares two text files and returns the similarity scores and tense/voice changes.
        @args: file1 (str): The path to the first file
                file2 (str): The path to the second file
        @returns: tuple: A tuple containing the TF-IDF similarity, semantic similarity, tense change, and voice change
    '''
    doc1 = read_and_preprocess(file1)
    doc2 = read_and_preprocess(file2)
    similarity_tfidf = calculate_similarity(doc1, doc2)
    similarity_semantic = calculate_semantic_similarity(doc1, doc2)
    tense_change, voice_change = compare_tense_and_voice(doc1, doc2)
    return similarity_tfidf, similarity_semantic, tense_change, voice_change

def compare_many_to_many(files1, files2):
    '''
        Compares multiple files to multiple files and returns the comparison results.
        @args: files1 (list): A list of file paths for the first set of files
                files2 (list): A list of file paths for the second set of files
        @returns: dict: A dictionary containing the comparison results
    '''
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

#----------------- Data visualization functions -----------------#
def plot_similarity_scores(similarity_scores_tfidf, similarity_scores_semantic):
    '''
        Plots the distribution of similarity scores for TF-IDF and semantic similarity.
        @args: similarity_scores_tfidf (list): A list of TF-IDF similarity scores
                similarity_scores_semantic (list): A list of semantic similarity scores
        @returns: None
    '''
    plt.figure(figsize=(10, 5))
    plt.hist(similarity_scores_tfidf, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='TF-IDF')
    plt.hist(similarity_scores_semantic, bins=20, color='orange', edgecolor='black', alpha=0.7, label='Semantic')
    plt.xlabel('Similarity Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred):
    '''
        Plots the confusion matrix for the plagiarism detection results.
        @args: y_true (list): A list of true labels
                y_pred (list): A list of predicted labels
        @returns: None
    '''
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
def plot_roc_curve(y_true, y_pred):
    '''
        Plots the ROC curve for the plagiarism detection results.
        @args: y_true (list): A list of true labels
                y_pred (list): A list of predicted labels
        @returns: None
    '''
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

#----------------- User interface function -----------------#
def user_interface():
    '''
        Displays a user interface for selecting files and comparing them.
        @returns: None
    '''
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

    write_results_to_file('./Results/results.txt', results)
    generate_summary_report(results)
    
    if similarity_scores_tfidf and similarity_scores_semantic:
        plot_similarity_scores(similarity_scores_tfidf, similarity_scores_semantic)
        y_true = [categorize_plagiarism(score) for score in similarity_scores_tfidf]
        y_pred = [categorize_plagiarism(score) for score in similarity_scores_semantic]
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_pred)
        
    if choice == 3 or choice == 4 or choice == 2:
        save_results_to_csv(results, './Results/results.csv')
        save_results_to_md(results, './Results/results.md')
        convert_to_pdf('./Results/results.md', './Results/results.pdf')
        convert_to_html('./Results/results.md', './Results/results.html')
        tk.messagebox.showinfo("Results", "The results have been saved successfully on Results directory.")

    else:
        file_path1 = './Results/results.csv'
        file_path2 = './Results/results.md'
        file_path3 = './Results/results.pdf'
        file_path4 = './Results/results.html'
        if os.path.exists(file_path1):
            os.remove(file_path1)
        if os.path.exists(file_path2):
            os.remove(file_path2)
        if os.path.exists(file_path3):
            os.remove(file_path3)
        if os.path.exists(file_path4):
            os.remove(file_path4)
        tk.messagebox.showinfo("Results", "The results have been saved successfully on Results directory ONLY TXT.")    
    
    # ask to open the directory of results on tkinter
    open_results = tk.messagebox.askyesno("Open Results", "Do you want to open the directory of results? (Only on WINDOWS)")
    if open_results:
        current_path = os.getcwd()
        subprocess.run(f'explorer {current_path}\\Results')
    

#----------------- Main function -----------------#
if __name__ == '__main__':
    user_interface()