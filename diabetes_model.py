# diabetes_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib

df = pd.read_csv('diabetes.csv')
x = df.drop(columns='Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=2)

model = svm.SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)

joblib.dump(model, 'svm_model.sav')
joblib.dump(scaler, 'scaler.sav')
