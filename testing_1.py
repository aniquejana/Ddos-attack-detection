import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

x=['RandomForestClassifier()','LogisticRegression() ','KNeighborsClassifier()','DecisionTreeClassifier()','GaussianNB()']
list_accuracy=[99.96120265036504, 98.88128685748215, 99.94703318354182, 99.95850370430347, 37.628705990310785]

harvest = np.array(
    [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [ 0, 0, 0, 0, 1]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x)), labels=x)
ax.set_yticks(np.arange(len(list_accuracy)), labels=list_accuracy)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(list_accuracy)):
    for j in range(len(x)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="green")

ax.set_title("Heat Diagram of DDOS attack detection (in accuracy(%)/ML Classifiers)")
fig.tight_layout()
plt.show()