#modules import
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import csv
from sklearn.svm import LinearSVC

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



import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

stroke = pd.read_csv('E:\\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv')
test_stroke = pd.read_csv('E:\\Reseach_Work_on_DDOS\CSV_file\kddcup99_csv.csv')
print(stroke.head())

#strokes = stroke.drop('id', axis=1)
print(stroke.select_dtypes(include='object')[:5])

# map normal to 0, all attacks to 1
is_attack = stroke.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_stroke.attack.map(lambda a: 0 if a == 'normal' else 1)

#data_with_attack = df.join(is_attack, rsuffix='_flag')
stroke['attack_flag'] = is_attack
test_stroke['attack_flag'] = test_attack

# view the result
print("Dataset of attack_flag")
print(stroke.head())



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
attack_map = stroke.attack.apply(map_attack)
stroke['attack_map'] = attack_map

test_attack_map = test_stroke.attack.apply(map_attack)
test_stroke['attack_map'] = test_attack_map

# view the result
print("Dataset of attack_map")
print(stroke.head())

# attack vs Multiple Classifier System protocols
attack_vs_protocol = pd.crosstab(stroke.attack, stroke.protocol_type)

print(attack_vs_protocol)

# attack_flag vs attack_map 
attack_flag_vs_attack_map = pd.crosstab(stroke.attack_flag, stroke.attack_map)
print("\nattack_flag_vs_attack_map\n")

print(attack_flag_vs_attack_map)

#df_utils=df.drop('protocol_type',axis=1)
features_to_encode = ['protocol_type', 'service', 'flag']
df_utils = pd.get_dummies(stroke[features_to_encode])
test_df_utils = pd.get_dummies(test_stroke[features_to_encode])

# not all of the features are in the test set, so we need to account for diffs
test_index = np.arange(len(test_stroke.index))
column_diffs = list(set(df_utils.columns.values)-set(test_df_utils.columns.values))

Data=pd.DataFrame(0, index=test_index, columns=column_diffs)
column_order = df_utils.columns.to_list()
test_encoded_temp = test_df_utils.join(Data)


# reorder the columns
test_final = test_encoded_temp[column_order].fillna(0)

# get numeric features, we won't worry about encoding these at this point
numeric_features = ['duration', 'src_bytes', 'dst_bytes']


# model to fit/test
to_fit = df_utils.join(stroke[numeric_features])
test_set = test_final.join(test_stroke[numeric_features])
print(to_fit)
print(test_set)

# create our target classifications
binary_y = stroke['attack_flag']
multi_y = stroke['attack_map']
print(binary_y)


test_binary_y = test_stroke['attack_flag']
test_multi_y = test_stroke['attack_map']

# build the training sets
#Splitting the dataset
binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(to_fit, stroke['attack_flag'],random_state=0, test_size=0.6)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(to_fit, test_stroke['attack_map'],random_state=0, test_size = 0.6)


#To Standardize the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
binary_train_X= sc_X.fit_transform(binary_train_X)
binary_val_X=sc_X.transform(binary_val_X)
multi_train_X=sc_X.fit_transform(multi_train_X)
multi_val_X=sc_X.transform(multi_val_X)

classifier = SVC()
classifier.fit(binary_train_X, binary_train_y)

y_pred = classifier.predict(binary_val_X)


cm = confusion_matrix(binary_val_y, y_pred)
print(cm)

acc = accuracy_score(binary_val_y, y_pred)
print(acc)