
#DDOS ATTACK DETECTION BY USING ENSEMBLE LEARNING WITHOUT USING INDEPENDENT COMPONENT ANALYSIS(ICA)
#ML CLASSIFIERS ARE USED:
#    
#      1. RandomForestClassifier          
#      2. LogisticRegression
#      3. KNeighborsClassifier
#      4. DecisionTreeClassifier
#      5. GaussianNB


#modules import
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import csv


# model imports
from sklearn.svm import LinearSVC,SVC
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler



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
from sklearn.metrics import RocCurveDisplay,DetCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc






# fetch the training file

file_path_full_training_set = 'E:\\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv'

file_path_test = 'E:\\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv'

#df = pd.read_csv()
df = pd.read_csv(file_path_full_training_set,low_memory=False)
test_df = pd.read_csv(file_path_test,low_memory=False)

print(df)




# sanity check
#print(df.head())
#print(test_df.head())


# map normal to 0, all attacks to 1
is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)

#data_with_attack = df.join(is_attack, rsuffix='_flag')
df['attack_flag'] = is_attack
test_df['attack_flag'] = test_attack

# view the result
print("Dataset of attack_flag")
print(df.head())



# lists to hold our attack classifications
dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
U2R = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
Sybil = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

  
# we will use these for plotting below
attack_labels = ['Normal','DoS','Probe','U2R','Sybil']


# helper function to pass to data frame mapping
def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in U2R:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in Sybil:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0
        
    return attack_type


# map the data and join to the data set
attack_map = df.attack.apply(map_attack)
df['attack_map'] = attack_map

test_attack_map = test_df.attack.apply(map_attack)
test_df['attack_map'] = test_attack_map

# view the result
print("Dataset of attack_map")
print(df.head())

# attack vs Multiple Classifier System protocols
attack_vs_protocol = pd.crosstab(df.attack, df.protocol_type)

print(attack_vs_protocol)

# attack_flag vs attack_map 
attack_flag_vs_attack_map = pd.crosstab(df.attack_flag, df.attack_map)
print("\nattack_flag_vs_attack_map\n")

print(attack_flag_vs_attack_map)

# helper function for drawing mulitple charts.
def bake_pies(data_list,labels):
    list_length = len(data_list)
    
    # setup for mapping colors
    color_list = sns.color_palette()
    color_cycle = itertools.cycle(color_list)
    cdict = {}
    # build the subplots
    fig, axs = plt.subplots(1, list_length,figsize=(18,10), tight_layout=False)
    plt.subplots_adjust(wspace=1/list_length)
    # loop through the data sets and build the charts
    for count, data_set in enumerate(data_list): 
        
        # update our color mapt with new values
        for num, value in enumerate(np.unique(data_set.index)):
            if value not in cdict:
                cdict[value] = next(color_cycle)
        # build the wedges
        wedges,texts = axs[count].pie(data_set,
                           colors=[cdict[v] for v in data_set.index])

        # build the legend
        axs[count].legend(wedges, data_set.index,
                           title="ATTACK",
                           loc="center left",
                           bbox_to_anchor=(-0.5, 0, 0.5, 1))
        # set the title
        axs[count].set_title(labels[count])
        
    return axs  



# get the series for each protocol
icmp_attacks = attack_vs_protocol.icmp
tcp_attacks = attack_vs_protocol.tcp
udp_attacks = attack_vs_protocol.udp

# create the charts
bake_pies([icmp_attacks, tcp_attacks, udp_attacks],['ICMP Protocol Type','TCP Protocol Type','UDP Protocol Type'])
plt.show()

# get a series with the count of each flag for attack and normal traffic
normal_flags = df.loc[df.attack_flag == 0].flag.value_counts()
attack_flags = df.loc[df.attack_flag == 1].flag.value_counts()

# create the pie charts
flag_axs = bake_pies([normal_flags, attack_flags], ['normal','attack'])        
plt.show()

# get a series with the count of each service for attack and normal MCS
normal_services = df.loc[df.attack_flag == 0].service.value_counts()
attack_services = df.loc[df.attack_flag == 1].service.value_counts()

# create the charts
service_axs = bake_pies([normal_services, attack_services], ['normalMCS','attack'])        
plt.show()

#Feature Engineering
# get the intial set of encoded features and encode them
features_to_encode = ['protocol_type', 'service', 'flag']
encoded = pd.get_dummies(df[features_to_encode])
test_encoded_base = pd.get_dummies(test_df[features_to_encode])

# not all of the features are in the test set, so we need to account for diffs
test_index = np.arange(len(test_df.index))
column_diffs = list(set(encoded.columns.values)-set(test_encoded_base.columns.values))

diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

# we'll also need to reorder the columns to match, so let's get those
column_order = encoded.columns.to_list()
# append the new columns
test_encoded_temp = test_encoded_base.join(diff_df)

# reorder the columns
test_final = test_encoded_temp[column_order].fillna(0)

# get numeric features, we won't worry about encoding these at this point
numeric_features = ['duration', 'src_bytes', 'dst_bytes']


# model to fit/test
to_fit = encoded.join(df[numeric_features])
test_set = test_final.join(test_df[numeric_features])
print(to_fit)
print(test_set)

# create our target classifications
binary_y = df['attack_flag']
multi_y = df['attack_map']
print(binary_y)

test_binary_y = test_df['attack_flag']
test_multi_y = test_df['attack_map']

# build the training sets
#Splitting the dataset
binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(test_set, binary_y,random_state=0, test_size=0.3)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(test_set, multi_y,random_state=0, test_size = 0.3)

#To Standardize the data
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
from sklearn import preprocessing
scaling=MinMaxScaler(feature_range=(-1,1)).fit(binary_train_X)
binary_train_X= scaling.transform(binary_train_X)
binary_val_X=scaling.transform(binary_val_X)
binary_train_X= preprocessing.scale(binary_train_X)
binary_val_X=preprocessing.scale(binary_val_X)

#binary_train_X= sc_X.fit_transform(binary_train_X)
#binary_val_X=sc_X.transform(binary_val_X)
scaling=MinMaxScaler(feature_range=(-1,1)).fit(multi_train_X)
#multi_train_X=sc_X.fit_transform(multi_train_X)
#multi_val_X=sc_X.transform(multi_val_X)
multi_train_X=scaling.transform(multi_train_X)
multi_val_X=scaling.transform(multi_val_X)




x=['SVM(Non-Linear)','SVM(Linear)','RandomForestClassifier()','LogisticRegression() ',' KNeighborsClassifier()','DecisionTreeClassifier()','GaussianNB()']

list_accuracy=[]
models=[]
models =[ 
        
        SVC(kernel='rbf',max_iter=4,probability=True),
        SVC(kernel='linear',max_iter=4,probability=True),
        #RandomForestClassifier(),
        #LogisticRegression(max_iter=494021),
        
        #KNeighborsClassifier(),
        #DecisionTreeClassifier(),
        #GaussianNB()
        ]
for model in models:

    binary_model = model
    binary_model.fit(binary_train_X, binary_train_y)
    binary_predictions = binary_model.predict(binary_val_X)



    
    #To get the Precision-Recall curve
    # predict probabilities
    lr_probs = model.predict_proba(binary_val_X)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = model.predict(binary_val_X)
    lr_precision, lr_recall, _ = precision_recall_curve(binary_val_y, lr_probs)
    lr_f1, lr_auc = f1_score(binary_val_y, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print(model, ': f1_Score=%.10f  and  auc=%.10f' % (lr_f1, lr_auc))

    # plot the precision-recall curves
    no_skill = len(binary_val_y[binary_val_y==1]) / len(binary_val_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label=model)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
  
    # calculate and display our base accuracty
    base_rf_score = accuracy_score(binary_val_y,binary_predictions)
    print( model, "Accuracy Score" ,(base_rf_score*100),"%")
    list_accuracy.append((base_rf_score*100))



    
    #To get the Confusion Matrix
    confusion_mat =confusion_matrix(binary_val_y,binary_predictions)
    print("Confusion Matrix:  ")
    print(confusion_mat)

    #To visualize Confusion Matrix
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(confusion_mat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(x=j, y=i,s=confusion_mat[i, j], va='center', ha='center', size='xx-large')
 
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    
    #To get DET curve
    fig, ax_det = plt.subplots(figsize=(7.5, 7.5))
    DetCurveDisplay.from_estimator(binary_model, binary_val_X, binary_val_y, ax=ax_det, name=model)
    ax_det.set_title("Detection Error Tradeoff (DET) curves")
    ax_det.grid(linestyle="--")
    plt.legend()
    plt.show()

    
    #View the classification report for test data and prediction
    print("Classification report")
    print(classification_report(binary_val_y,binary_predictions))
    
    #ROC Curve 
    clf = model
    clf.fit(binary_train_X,binary_train_y)
    RocCurveDisplay.from_estimator(clf,binary_val_X,binary_val_y)
    plt.show()

    #auc_scores
    auc_score1=roc_auc_score(binary_val_y,binary_predictions)
    print("Area Under thr ROC curve",auc_score1)








print("List of ML Classifiers:",models)
print("List of the accuracy:",list_accuracy) 

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]//2, y[i], ha = 'center',color='white',rotation='vertical')

plt.bar(x,list_accuracy,color=['green','blue','red','purple','black','yellow','pink'])
addlabels(x, list_accuracy)
plt.xlabel('Categories')
plt.ylabel("Accuracy Values(%)")
plt.title('Bar Plot of Accuracy')
plt.show()

  
#To get Heat Maps
harvest = np.array(
    [
    [1,0,0,0,0,0,0],
    [0, 1, 0, 0, 0,0,0],
    [0, 0, 1, 0, 0,0,0],
    [0, 0, 0, 1, 0,0,0],
    [0, 0, 0, 0, 1,0,0],
    [ 0, 0, 0, 0,0,1, 0],
    [0,0,0,0,0,0,1]
    ])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(x)), labels=x)
ax.set_yticks(np.arange(len(list_accuracy)), labels=list_accuracy)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(list_accuracy)):
    for j in range(len(x)):
        text = ax.text(j, i, harvest[i, j],ha="center", va="center", color="green")

ax.set_title("Heat Diagram of DDOS attack detection (in accuracy(%)/ML Classifiers)")
fig.tight_layout()
plt.show()









