## What is cognitive science?
R = Cognitive science is the interdisciplinary study of the mind
and intelligence, embracing philosophy, psychology,
artificial intelligence, neuroscience, linguistics, and
anthropology

### Other definitions?
R = Information processing

## What is AI?
R = Study of mental faculties through the use of computational models

## What is ML?
R =  Ability to use percepts from the outside world 
- “Machine learning uses data and answers to discover rules behind a problem"
- “Machine learning is programming computers to optimize a performance criterion using example data or past experience.”

## Unsupervised learning
-  Learning patterns in the input when no specific output values are
supplied.
-   Learning patterns in the input when no specific output values are supplied.

### Clustering
- Centroid-based
- Density-based
- Distribution based
- Hierarchical

#### Similarity measures

- Euclidian Distance
- Manhattan Distance
- Minkowski Distance
- Hamming Distance
- Chebyshev Distance
- Levenshtein Distance
- Cosine Distance

#### K-means

Definition: Clustering method that tries to minimize the distance between points in a cluster (inertia)
- Centroid-based algorithm
- Distance-based algorithm


__Examples__:
- Learn to separate colors.
- Learn when it might rain.
- Learn how to detect people that will not pay their credit cards.

## Semisupervised learning

Definition: Supervised learning is the process of discovering a function h (hypothesis) that approximates a real unknown function f

Some data is labelled – usually a very small part
 
__Learner learns to__:
- Generate labelled data and to
- Detect regularities in the input 


## Creating a dataset

All the Y column is called a Feature

And a row is called an instance

|  x |y   | z  | 
|---|---|---|
|1132   | 1451435  |  23423 |
| 134134  |  545 |  34576 |
|  15425 |  543354 | 524534  |


## Binary confusion matrix

| Positive  |  Negative |
|---|---|
| TP  | FP  |
| FN  |  TN |


_True Positive (TP)_ 
- Predicted value matches actual
- Both were positive

_True Negative (TN)_
- Predicted value matches actual
- Both are negative

_False Positive (FP)_
- Type I error
- Predicted value falsely predicted
- Actual value Negative

_False Negative (FN)_
- Type II error
- Predicted value falsely predicted
- Actual value Positive

### Confusion Matrix Metrix

__Accuracy__
- Fraction of predictions model correctly classified

Accuracy = Corrected predictions / Total number of predictions

For binary classification

Accuracy = (TP+TN) / (TP+TN+FP+FN)

__Precision__
- Proportion predicted positives identified correctly

Precision = TP / TP+FP

__Recall (Sensitivity)__
- Proportion actual positives identified correctly

Recall = TP / TP+FN

__Specificity__
- Proportion actual negatives identified correctly

Specificity = TN / TN+FP

## Which attribute is the best classifier?
- Information gain
    - Measures how well a given attribute separates training examples according to their classification.
- Entropy
    - Characterizes (im)purity of an arbitrary collection of examples.
    - Given a collection of examples (S), containing + and – examples:

## Support Vector Machines (SVM)

Definition: One of the most popular algorithms for supervised learning

### SVMs
- Main problem: which line?
    - Many lines (hyperplanes) can be selected to divide the two classes.
    - SVMs select the hyperplane that has the maximum distance (margin) between data points of classes

### Kernel SVM
Used when data is non-linearly separable
Data is transformed to a higher dimension space
In the new dimension space, a linear hyperplane can be found