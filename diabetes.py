import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score  
df = pd.read_csv(r'C:\Users\Ajay kumar\Downloads\diabetes.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df['Outcome'].value_counts())
print(df.groupby('Outcome').mean())
x = df.drop(columns='Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
scaler.fit(x)
Standardized_data = scaler.transform(x)
print(Standardized_data)
x = Standardized_data
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy score of the training data = ', training_data_accuracy)
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy score of the test data = ', test_data_accuracy)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
columns = df.drop(columns='Outcome').columns
input_data_reshaped_df = pd.DataFrame(input_data_reshaped, columns=columns)
std_data = scaler.transform(input_data_reshaped_df)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
