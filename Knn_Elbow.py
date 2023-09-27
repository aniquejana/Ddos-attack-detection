#ules import
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import csv
#import tensorflow as tf

# model imports
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor


# processing imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score



import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt 

import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("E:\\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv")

print(df.head())

print(df.isnull().sum())

print(df.info())

print(df.describe())

print(df['duration'].unique())

print(df['duration'].value_counts())

#sns.countplot(data=df, x='duration')

print(df.corr()['duration'].sort_values(ascending=False))

X = df.drop(['duration'], axis=1)
y = df['duration']

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,plot_confusion_matrix
knn_model = KNeighborsClassifier()

test_error_rates = []

"""
for k in range(1,40):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train,y_train) 
   
    y_pred = knn_model.predict(X_test)
    
    test_error = 1 - accuracy_score(y_test,y_pred)
    test_error_rates.append(test_error)

plt.figure(figsize=(6,4),dpi=100)
plt.plot(range(1,40),test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel("K Value")  

knn_model = KNeighborsClassifier(n_neighbors=32)
knn_model.fit(X_train,y_train)
KNeighborsClassifier(n_neighbors=32)
y_pred = knn_model.predict(X_test)
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(knn_model, X_test, y_test)
print(classification_report(y_test, y_pred))

"""