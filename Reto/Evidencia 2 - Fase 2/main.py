import os
import re
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from sklearn.metrics import pairwise, roc_auc_score
from tkinter import filedialog, Tk
import numpy as np

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def read_file(file_path):
    """Reads a file and returns its content."""
    with open(file_path, 'r') as file:
        return file.read()

def bert_embed(text):
    """Generate BERT embeddings for the given text."""
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        output = model(**encoded_input)
    return output.last_hidden_state[:,0,:].numpy()

def compute_similarity(embed1, embed2):
    """Computes the cosine similarity between two embeddings."""
    return pairwise.cosine_similarity(embed1, embed2)[0, 0]

def check_plagiarism(file1, file2):
    """Checks plagiarism between two files."""
    text1 = read_file(file1)
    text2 = read_file(file2)
    embed1 = bert_embed(text1)
    embed2 = bert_embed(text2)
    return compute_similarity(embed1, embed2)

def select_files(title="Select Files", filetypes=(("Text files", "*.txt"), ("All files", "*.*"))):
    """Allows user to select multiple files via a GUI dialog."""
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    root.destroy()
    return file_paths

def plot_histogram(similarities):
    """Plot a histogram of the similarity scores."""
    plt.hist(similarities, bins=10, color='blue', alpha=0.7)
    plt.title('Histogram of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.show()

def write_results(file_path, results):
    """Writes the plagiarism check results to a file."""
    with open(file_path, 'w') as file:
        for key, value in results.items():
            if isinstance(key, tuple):
                file.write(f'{key[0]} vs {key[1]}: {value:.2f}%\n')
            else:
                file.write(f'{key}: {value:.2f}%\n')

def perform_plagiarism_checks():
    """Performs various types of plagiarism checks based on user input."""
    print("Select the type of plagiarism check:")
    print("1: One file to one file")
    print("2: One file to multiple files")
    print("3: Multiple files to multiple files")
    print("4: Directory check")
    choice = input("Enter your choice (1-4): ")
    results = {}
    
    if choice == '1':
        files = select_files()
        similarity = check_plagiarism(files[0], files[1])
        results[(os.path.basename(files[0]), os.path.basename(files[1]))] = similarity

    elif choice == '2':
        file1 = select_files()[0]
        files = select_files()
        for file in files:
            similarity = check_plagiarism(file1, file)
            results[(os.path.basename(file1), os.path.basename(file))] = similarity

    elif choice == '3':
        files1 = select_files()
        files2 = select_files()
        for f1 in files1:
            for f2 in files2:
                similarity = check_plagiarism(f1, f2)
                results[(os.path.basename(f1), os.path.basename(f2))] = similarity

    elif choice == '4':
        directory = filedialog.askdirectory()
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
        for i, f1 in enumerate(files):
            for f2 in files[i+1:]:
                similarity = check_plagiarism(f1, f2)
                results[(os.path.basename(f1), os.path.basename(f2))] = similarity

    write_results('results.txt', results)
    plot_histogram(list(results.values()))

def main():
    """Main function to run the plagiarism checks."""
    perform_plagiarism_checks()

if __name__ == '__main__':
    main()
