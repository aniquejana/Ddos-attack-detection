import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]//2, y[i], ha = 'center',color='white')
x=['RandomForestClassifier()','LogisticRegression() ','KNeighborsClassifier()','DecisionTreeClassifier()','GaussianNB()']


list_accuracy=[99.96120265036504, 98.88128685748215, 99.94703318354182, 99.95850370430347, 37.628705990310785]
plt.bar(x,list_accuracy,color=['green','blue','red','purple','tomato'])
addlabels(x, list_accuracy)
plt.xlabel('Categories')
plt.ylabel("Accuracy Values(%)")
plt.title('Bar Plot of Accuracy')
plt.show()




