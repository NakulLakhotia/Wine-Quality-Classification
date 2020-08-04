import pandas as pd
import seaborn as sns   # for graphs and visualizations
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
 # % matplotlib inline  -- to be used in jupyter notebook

wine=pd.read_csv("winequality-red.csv",sep=';')
#print(wine.head())

#print(wine.info())
#print(wine.isnull().sum())
''' Pre-processing data'''
bins=(2,6.5,8)
group_names=['bad','good']
wine['quality']=pd.cut(wine['quality'],bins=bins,labels=group_names)
#print(wine['quality'].unique)
label_quality=LabelEncoder()
wine['quality']=label_quality.fit_transform(wine['quality'])
#print(wine.head())
#print(wine['quality'].value_counts())
#print(sns.countplot(wine['quality']))

''' Separate dataset as response variable and feature variables'''

X=wine.drop('quality',axis=1) # all variables except quality variable forms the feature variables
y=wine['quality']
# Train,test and split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Apply standard scaling to get optimized result

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

''' RANDOM FOREST CLASSIFIER '''
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)    # fit the training data into the model
pred_rfc=rfc.predict(X_test)   # predict using the test data
#print(pred_rfc[:10])

''' checking the model performance '''
print("Classification Report:\n",classification_report(y_test,pred_rfc))   # y_test v/s pred_rfc
print("Confusion Matrix is:\n", confusion_matrix(y_test,pred_rfc))
print("Accuracy Score is:\n",accuracy_score(y_test,pred_rfc))

''' SVM Classifier '''
svmc=svm.SVC()
svmc.fit(X_train,y_train)
pred_svmc=svmc.predict(X_test)
print("Classification Report:\n",classification_report(y_test,pred_svmc))   # y_test v/s pred_rfc
print("Confusion Matrix is:\n", confusion_matrix(y_test,pred_svmc))

''' MULTILAYER PERCEPTRON CLASSIFIER '''
mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(X_train,y_train)
pred_mlpc=mlpc.predict(X_test)
print("Classification Report:\n",classification_report(y_test,pred_mlpc))   # y_test v/s pred_rfc
print("Confusion Matrix is:\n", confusion_matrix(y_test,pred_mlpc))

''' Prediction of a new dataset '''
Xnew=[[7.3,0.58,0.00,2.0,0.065,15.0,21.0,0.9946,3.36,0.47,10.0]]
Xnew=sc.transform(Xnew)
ynew=rfc.predict(Xnew)
print("\n\nPredicted label:",ynew,"\n")    #the output is 0, meaning its a bad wine

