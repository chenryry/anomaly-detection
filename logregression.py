import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('creditcard.csv')
df.isna().sum()
columns = df.columns.tolist()
target = "Class"
X = df[columns]
Y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=21)

smote = SMOTE(random_state=21)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],                   
    'max_iter': [100, 200, 500,1000]                 
}

logreg = LogisticRegression(random_state=21, class_weight="balanced")
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='f1', verbose=1)
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)

print(accuracy_score(y_pred,y_test))
print(f1_score(y_pred,y_test))