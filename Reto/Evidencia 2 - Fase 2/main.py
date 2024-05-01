import os
import re
import nltk
import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise

# Ensure necessary nltk resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def read_file(file_path):
    """Reads a file and returns its content."""
    with open(file_path, 'r') as file:
        return file.read()

def preprocess_text(text):
    """Preprocesses the text by removing non-alphanumeric characters and lowercasing."""
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.lower()
    return text

def tokenize_and_filter(text):
    """Tokenizes the text and filters out stopwords."""
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = nltk.word_tokenize(text)
    return [word for word in words if word not in stop_words]

def stem_and_lemmatize(words):
    """Applies stemming and lemmatization to a list of words."""
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(stemmer.stem(word)) for word in words])

def compute_similarity(text1, text2):
    """Computes the cosine similarity between two pieces of text."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return pairwise.cosine_similarity(vectors)[0, 1]

def check_plagiarism(file1, file2):
    """Checks plagiarism between two files."""
    text1 = stem_and_lemmatize(tokenize_and_filter(preprocess_text(read_file(file1))))
    text2 = stem_and_lemmatize(tokenize_and_filter(preprocess_text(read_file(file2))))
    return compute_similarity(text1, text2)

def select_files(title="Select Files", filetypes=(("Text files", "*.txt"), ("All files", "*.*"))):
    """Allows user to select multiple files via a GUI dialog."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    root.destroy()
    return file_paths

def select_directory():
    """Allows user to select a directory via a GUI dialog."""
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    root.destroy()
    return directory

def perform_plagiarism_checks():
    """Performs various types of plagiarism checks based on user input."""
    print("Select the type of plagiarism check:")
    print("1: One file to one file")
    print("2: One file to multiple files")
    print("3: Multiple files to multiple files")
    print("4: Directory check")
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        file1 = select_files()[0]
        file2 = select_files()[0]
        similarity = check_plagiarism(file1, file2)
        print(f'Similarity: {similarity:.2f}')
        write_results('results.txt', {'One-to-One Similarity': similarity})
    
    elif choice == '2':
        file1 = select_files()[0]
        files = select_files()
        similarities = {os.path.basename(f): check_plagiarism(file1, f) for f in files}
        write_results('results.txt', similarities)
    
    elif choice == '3':
        files1 = select_files()
        files2 = select_files()
        similarities = {(os.path.basename(f1), os.path.basename(f2)): check_plagiarism(f1, f2) 
                        for f1 in files1 for f2 in files2}
        write_results('results.txt', similarities)
    
    elif choice == '4':
        directory = select_directory()
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
        similarities = {(os.path.basename(f1), os.path.basename(f2)): check_plagiarism(f1, f2) 
                        for i, f1 in enumerate(files) for f2 in files[i + 1:]}
        write_results('results.txt', similarities)

def write_results(file_path, results):
    """Writes the plagiarism check results to a file."""
    with open(file_path, 'w') as file:
        for key, value in results.items():
            if isinstance(key, tuple):
                value *= 100
                file.write(f'{key[0]} vs {key[1]}: {value:.2f}% \n')
            else:
                value *= 100
                file.write(f'{key}: {value:.2f}%\n')

def main():
    """Main function to run the plagiarism checks."""
    perform_plagiarism_checks()

if __name__ == '__main__':
    main()
