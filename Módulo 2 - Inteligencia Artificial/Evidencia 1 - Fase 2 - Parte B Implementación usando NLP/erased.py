# Function to check the performance of the model through the AUC-ROC curve, True Positive Rate, and False Positive Rate, False Negative Rate and True Negative Rate, and the confusion matrix
def check_performance(y_true, y_pred):
    '''
    Function to check the performance of the model

    Args:
        y_true (list): The list of true values
        y_pred (list): The list of predicted values
    '''
    # Computing the AUC-ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
    auc = sklearn.metrics.auc(fpr, tpr)
    
    # Computing the confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    # Writing the confusion matrix to the results.txt file
    with open('results.txt', 'a') as file:
        file.write('Confusion Matrix\n')
        file.write(str(cm))
    
    # Computing the True Positive Rate, False Positive Rate, False Negative Rate, and True Negative Rate
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tn)
    tnr = tn / (tn + fp)
    # Writing the performance metrics to the results.txt file
    with open('results.txt', 'a') as file:
        file.write('True Positive Rate: {}\n'.format(tpr))
        file.write('False Positive Rate: {}\n'.format(fpr))
        file.write('False Negative Rate: {}\n'.format(fnr))
        file.write('True Negative Rate: {}\n'.format(tnr))
        
    return auc, cm, tpr, fpr, fnr, tnr

# Plotting the confusion matrix, True Positive Rate, False Positive Rate, False Negative Rate, and True Negative Rate, and the AUC-ROC curve, plagiarism score and the similarity score
def plot_results(y_true, y_pred, similarity_scores):
    '''
    Function to plot the results

    Args:
        y_true (list): The list of true values
        y_pred (list): The list of predicted values
        similarity_scores (list): The list of similarity scores
    '''
    # Plotting the AUC-ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
    auc = sklearn.metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # Plotting the confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    plt.matshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Plotting the plagiarism scores
    plt.plot(similarity_scores)
    plt.xlabel('Pair of Files')
    plt.ylabel('Plagiarism Score')
    plt.title('Plagiarism Scores')
    plt.show()
    
    # Plotting the similarity scores
    plt.plot(similarity_scores)
    plt.xlabel('Pair of Files')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Scores')
    plt.show()
        
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
    
        # Checking plagiarism in a directory and getting similarity scores and true labels
    similarity_scores, true_labels = check_plagiarism_directory(directory)
    
    # Set a threshold
    threshold = 0.
    
    # Calculate y_pred based on the threshold
    y_pred = [1 if score >= threshold else 0 for score in similarity_scores]
    
    # Calculate performance metrics
    auc, cm, tpr, fpr, fnr, tnr = check_performance(true_labels, y_pred)
    
    # Plot the results
    plot_results(true_labels, y_pred, similarity_scores)
    
    # Print the AUC
    print('AUC:', auc)
    

# Checking if the script is run as the main script
if __name__ == '__main__':
    main()