import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
from sklearn.model_selection import train_test_split
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

df = pd.read_csv('creditcard.csv')

df.isnull().sum()

df1= df.sample(frac = 0.1,random_state=1)
Fraud = df1[df1['Class']==1]
Valid = df1[df1['Class']==0]
outlier_fraction = len(Fraud)/float(len(Valid))

df = df.drop("Time", axis =1)
#Create independent and Dependent Features
columns = df.columns.tolist()

# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)

X = df[columns]
Y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

model_IF = IsolationForest(n_estimators=100, contamination=outlier_fraction,random_state=42, verbose=0)

model_IF.fit(X_train)
scores_prediction = model_IF.decision_function(X_test)
y_pred = model_IF.predict(X_test)
binary_pred = [0 if p == 1 else 1 for p in y_pred]
print(accuracy_score(y_test,binary_pred))
print(f1_score(y_test,binary_pred))
print(precision_score(y_test,binary_pred))
print(recall_score(y_test,binary_pred))