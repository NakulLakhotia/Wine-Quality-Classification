# Wine Classification-Scikit-Learn
Python code to classify "wine quality" as good or bad- RandomForestClassifier, SVM Classifier, MLP Classifier

The csv format of the provided dataset was downloaded from the link:
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

This python code uses various libraries provided by python like pandas,seaborn,scikit-learn.
Scikit-learn is probably the most useful library for machine learning in Python. It is on NumPy, SciPy and matplotlib, this library contains a lot of effiecient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.

** Various code in the comments (#comment) can also be executed by simply removing the "#" in front of them.

** Details of dataset

Features---->fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol"

Response variable---->"quality"

** Steps involved

1) Import all the necessary libraries

2) Load the datset using pandas

2) Pre-processing the data
      2.1) Using pandas.cut we segment and sort data values into bins, labels are set as 'bad' or 'good'
      2.2) Using LabelEncoder we set the labels as 0-bad wine & 1-good wine

3) Now divide the dataset into response variable and feature variables

4) Scale the training data using StandardScaler 

5) Use the RandomForestClassifier to develop a machine learning model which trains the data

6) Predict using the test data

7) To check the model performance use classification_repot, confusion_matrix & accuracy_score

8) Repeat steps 5-7 for different classifiers like SVMClassifier & MLPClassifier

9) Later given is an example of the model predicting a new data using the RandomForestClassifier 
