'''
    @author: @Lunaespindola
    @date: 2024/04/16
    @description:
        This is the code to check plagiarism in a text file or multiple text files and returns a brief on a txt.
'''

# Importing the necessary libraries
import os
import sys
import re
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import filedialog

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
    with open(file_path, 'r', encoding='utf-8') as file:
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

# ------------------------------------------- Functions to check plagiarism -------------------------------------------
    
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
    return similarity_scores
    
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
    return similarity_scores
    
# Function to check plagiarism in a directory and return similarity scores and true labels
def check_plagiarism_directory(directory, threshold=0.7):
    '''
    Function to check plagiarism in a directory and return similarity scores and true labels

    Args:
        directory (str): The path of the directory
        threshold (float): The threshold for considering a pair as plagiarized

    Returns:
        list: The similarity scores
        list: The true labels
    '''
    files = os.listdir(directory)
    similarity_scores = []
    true_labels = []
    files_1 = []
    files_2 = []
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            file1 = os.path.join(directory, files[i])
            file2 = os.path.join(directory, files[j])
            similarity = check_plagiarism(file1, file2)
            similarity_scores.append(similarity)
            # Assuming you have a labeled dataset indicating whether each pair is plagiarized or not
            # Here, you need to replace this with your actual dataset
            true_label = 1 if files[i] == files[j] else 0  # Assuming plagiarism occurs if file names are the same
            true_labels.append(true_label)
            files_1.append(files[i])
            files_2.append(files[j]) 
            
    return similarity_scores, true_labels , files_1, files_2

# Function to select file
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()  # Destroy the tkinter window after file selection
    return file_path

# Function to select directory
def select_directory():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    root.destroy()  # Destroy the tkinter window after directory selection
    return directory

# ------------------------------------------- Functions to check the acuracity -------------------------------------------

# Function to check the accuracy
def check_accuracy(similarity_scores, true_labels, threshold=0.7):
    '''
    Function to check the accuracy

    Args:
        similarity_scores (list): The similarity scores
        true_labels (list): The true labels
        threshold (float): The threshold for considering a pair as plagiarized

    Returns:
        float: The accuracy
    '''
    predictions = [1 if score > threshold else 0 for score in similarity_scores]
    accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
    return accuracy

# ------------------------------------------- Functions to get the y_pred and y_true -------------------------------------------

# Function to get the y_pred and y_true
def get_y_pred_y_true(similarity_scores, threshold=0.7):
    '''
    Function to get the y_pred and y_true

    Args:
        similarity_scores (list): The similarity scores
        threshold (float): The threshold for considering a pair as plagiarized

    Returns:
        list: The predicted labels
        list: The true labels
    '''
    y_pred = [1 if score > threshold else 0 for score in similarity_scores]
    y_true = [1 if score > threshold else 0 for score in similarity_scores]
    return y_pred, y_true

# ------------------------------------------- Functions to get the performance metrics -------------------------------------------

# Function to get the performance metrics
def get_performance_metrics(y_true, y_pred):
    '''
    Function to get the performance metrics

    Args:
        y_true (list): The true labels
        y_pred (list): The predicted labels

    Returns:
        float: The AUC
        list: The confusion matrix
        float: The true positive rate
        float: The false positive rate
        float: The false negative rate
        float: The true negative rate
    '''
    # Check if y_true contains at least two unique classes
    unique_classes = set(y_true)
    if len(unique_classes) < 2:
        # If not, return default values and      a warning message
        print("Warning: There should be at least two unique classes in y_true.")
        return 0.5, [[0, 0], [0, 0]], 0.0, 0.0, 0.0, 0.0
    
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)
    return auc, cm, tpr, fpr, fnr, tnr

# ------------------------------------------- Functions to write the results to a file -------------------------------------------

# Function to write the results to a file
def write_results_to_file(file_path, results):
    '''
    Function to write the results to a file

    Args:
        file_path (str): The path of the file
        results (dict): The results to be written to the file
    '''
    with open(file_path, 'w') as file:
        for key, value in results.items():
            file.write(f'{key}: {value}\n')
            
# ------------------------------------------- Functions to plot the results -------------------------------------------


def plot_results(y_true, y_pred, similarity_scores):
    '''
    Function to plot the results

    Args:
        y_true (list): The true labels
        y_pred (list): The predicted labels
        similarity_scores (list): The similarity scores
    '''
    # Print out confusion matrix if there are two unique classes
    if len(set(y_true)) > 1:
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
    else:
        print("Cannot generate a meaningful confusion matrix due to the lack of diversity in the true labels.")

    # Plotting the AUC-ROC curve if there is variability in predictions
    if len(set(y_pred)) > 1:
        # Compute ROC curve and AUC if there is variability in predictions
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
        auc = sklearn.metrics.auc(fpr, tpr)

        # Plotting the ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print("Cannot generate a meaningful ROC curve due to the lack of variability in predictions.")

    # Similarity Scores Histogram
    plt.hist(similarity_scores, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Similarity Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Similarity Scores')
    plt.show()
    
    return

  

# ------------------------------------------- Functions to select files and function -------------------------------------------

# Function to select files
def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_paths = filedialog.askopenfilenames(
        title="Select Files", 
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )

    root.destroy()  # Destroy the root window after file selection

    return file_paths
# Function to select wich function to use
def select_function():
    '''
    Function to select wich function to use
    '''
    print('Select the function you want to use:')
    print('1. Check plagiarism between two files')
    print('2. Check plagiarism in a directory')
    print('3. Check plagiarism from one file to multiple files')
    print('4. Check plagiarism from multiple files to multiple files')
    print('5. Check plagiarism in a directory and return similarity scores and true labels')
    try :
        choice = int(input('Enter your choice: '))
    except ValueError:
        print('Only numbers from 1 to 5 are allowed as input')
        sys.exit(1)
        
    if choice == 1:
        file1 = select_file()
        file2 = select_file()
        similarity = check_plagiarism(file1, file2)
        accuracy = check_accuracy([similarity], [1])
        y_pred, y_true = get_y_pred_y_true([similarity], 0.5)
        auc, cm, tpr, fpr, fnr, tnr = get_performance_metrics(y_true, y_pred)
        results = {
            'Similarity Score': similarity,
            'Accuracy': accuracy,
            'AUC': auc,
            'True Positive Rate': tpr,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr,
            'True Negative Rate': tnr,
            'Confusion Matrix': 'Not available for single pair comparison'
        }
        write_results_to_file('Results.txt', results)
        plot_results(y_true, y_pred, [similarity])

    elif choice == 2:
        directory = select_directory()
        similarity_scores, true_labels, files_1, files_2 = check_plagiarism_directory(directory)
        accuracy = check_accuracy(similarity_scores, true_labels)
        y_pred, y_true = get_y_pred_y_true(similarity_scores)
        auc, cm, tpr, fpr, fnr, tnr = get_performance_metrics(y_true, y_pred)
        results = {
            'Similarity Scores': similarity_scores,
            'Accuracy': accuracy,
            'AUC': auc,
            'True Positive Rate': tpr,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr,
            'True Negative Rate': tnr,
            'Confusion Matrix': cm
        }
        write_results_to_file('Results.txt', results)
        plot_results(y_true, y_pred, similarity_scores)

    elif choice == 3:
        file_path = select_file()
        files_path = select_files() 
        similarity_scores = check_plagiarism_file_to_files(file_path, files_path)
        accuracy = check_accuracy(similarity_scores, [1]*len(similarity_scores))
        y_pred, y_true = get_y_pred_y_true(similarity_scores, 0.5)
        auc, cm, tpr, fpr, fnr, tnr = get_performance_metrics(y_true, y_pred)
        results = {
            'Similarity Scores': similarity_scores,
            'Accuracy': accuracy,
            'AUC': auc,
            'True Positive Rate': tpr,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr,
            'True Negative Rate': tnr,
            'Confusion Matrix': cm
            
        }
        write_results_to_file('Results.txt', results)
        plot_results(y_true, y_pred, similarity_scores)

    elif choice == 4:
        files1 = select_files()
        files2 = select_files()
        similarity_scores = check_plagiarism_files_to_files(files1, files2)
        accuracy = check_accuracy(similarity_scores, [1]*len(similarity_scores))
        y_pred, y_true = get_y_pred_y_true(similarity_scores, 0.5)
        auc, cm, tpr, fpr, fnr, tnr = get_performance_metrics(y_true, y_pred)
        results = {
            'Similarity Scores': similarity_scores,
            'Accuracy': accuracy,
            'AUC': auc,
            'True Positive Rate': tpr,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr,
            'True Negative Rate': tnr,
            'Confusion Matrix': cm
        }
        write_results_to_file('Results.txt', results)
        plot_results(y_true, y_pred, similarity_scores)

    elif choice == 5:
        directory = select_directory()
        similarity_scores, true_labels, files_1, files_2 = check_plagiarism_directory(directory)
        accuracy = check_accuracy(similarity_scores, true_labels)
        y_pred, y_true = get_y_pred_y_true(similarity_scores)
        auc, cm, tpr, fpr, fnr, tnr = get_performance_metrics(y_true, y_pred)
        results = {
            'Similarity Scores': similarity_scores,
            'Accuracy': accuracy,
            'AUC': auc,
            'True Positive Rate': tpr,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr,
            'True Negative Rate': tnr,
            'Confusion Matrix': cm
        }
        write_results_to_file('Results.txt', results)
        plot_results(y_true, y_pred, similarity_scores)
        
    else:
        print('Only numbers from 1 to 5 are allowed as input')

# ------------------------------------------- Main function -------------------------------------------

def main():
    select_function()

if __name__ == '__main__':
    main()