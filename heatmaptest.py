# import modules
import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb

# import file with data
data = pd.read_csv("E:\\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv")

# prints data that will be plotted
# columns shown here are selected by corr() since
# they are ideal for the plot
print(data.corr())

# plotting correlation heatmap
dataplot = sb.heatmap(data.corr(), cmap="YlGnBu", annot=True)

# displaying heatmap
mp.show()






import matplotlib.pyplot as plt
import numpy as np

#here's our data to plot, all normal Python lists
x=['RandomForestClassifier()','LogisticRegression() ','KNeighborsClassifier()','DecisionTreeClassifier()','GaussianNB()']
list_accuracy=[99.96120265036504, 98.88128685748215, 99.94703318354182, 99.95850370430347, 37.628705990310785]

intensity = [
    [5, 10, 15, 20, 25],
    [30, 35, 40, 45, 50],
    [55, 60, 65, 70, 75],
    [80, 85, 90, 95, 100],
    [105, 110, 115, 120, 125]
]

#setup the 2D grid with Numpy
x, y = np.meshgrid(x, list_accuracy)

#convert intensity (list of lists) to a numpy array for plotting
intensity = np.array(intensity)

#now just plug the data into pcolormesh, it's that easy!
plt.pcolormesh(x, y, intensity)
plt.colorbar() #need a colorbar to show the intensity scale
plt.show() #boom