#modules import
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
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
from sklearn.metrics import auc






# fetch the training file

file_path_full_training_set = 'E:\\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv'
# Path of the CSV file = "E:\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv"
file_path_test = 'E:\\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv'
#"C:\Users\ANIQUE JANA\OneDrive\Desktop\kddcup990_csv - Copy.csv"
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
#plt.show()

# get a series with the count of each flag for attack and normal traffic
normal_flags = df.loc[df.attack_flag == 0].flag.value_counts()
attack_flags = df.loc[df.attack_flag == 1].flag.value_counts()

# create the pie charts
flag_axs = bake_pies([normal_flags, attack_flags], ['normal','attack'])        
#plt.show()

# get a series with the count of each service for attack and normal MCS
normal_services = df.loc[df.attack_flag == 0].service.value_counts()
attack_services = df.loc[df.attack_flag == 1].service.value_counts()

# create the charts
service_axs = bake_pies([normal_services, attack_services], ['normalMCS','attack'])        
#plt.show()

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
binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(to_fit, binary_y,random_state=32, test_size=0.3)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(to_fit, multi_y,random_state=32, test_size = 0.3)

#To Standardize the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
binary_train_X= sc_X.fit_transform(binary_train_X)
binary_val_X=sc_X.transform(binary_val_X)
multi_train_X=sc_X.fit_transform(multi_train_X)
multi_val_X=sc_X.transform(multi_val_X)




#Calculating SVM() Accuracy
#dataset = load_digits()
#binary_train_X_1, binary_val_X_1, binary_train_y_1, binary_val_y_1 = train_test_split(dataset.data, dataset.target,random_state=0, test_size=0.30)
Classifier=SVC(kernel="linear",C=1.0)
Classifier.fit(binary_train_X,binary_train_y)
binary_predictions=Classifier.predict(binary_val_X)



    
accuracy=accuracy_score(binary_val_y,binary_predictions)*100
confusion_mat =confusion_matrix(binary_val_y,binary_predictions)
print("SVM() Accuracy Score :", accuracy)
print("confusion Matrix: ")
print(confusion_mat)
print("Classification report")
print(classification_report(binary_val_y,binary_predictions))


