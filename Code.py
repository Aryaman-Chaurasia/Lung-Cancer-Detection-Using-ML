
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import pyplot 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix 

print(" Dataset : ") 
dataset = pd.read_csv('Cancer Data.csv')
print(len(dataset)) 
print(dataset.head())

A = dataset[dataset.Result == 1] 
B = dataset[dataset.Result == 0]

plt.scatter(A.Age, A.Smokes, color="Green", label="1", alpha=0.4) 
plt.scatter(B.Age, B.Smokes, color="Orange", label="0", alpha=0.4) 
plt.xlabel(" Age ")
plt.ylabel(" Smokes ")
plt.legend()
plt.title(" Smokes vs Age ") 
plt.show()

plt.scatter(A.Age, A.Alkohol, color="Olive", label="1", alpha=0.4)
plt.scatter(B.Age, B.Alkohol, color="Blue", label="0", alpha=0.4) 
plt.xlabel(" Age ")
plt.ylabel(" Alcohol ")
plt.legend()
plt.title(" Alcohol vs Age ")
plt.show()

plt.scatter(A.Smokes, A.Alkohol, color="Red", label="1", alpha=0.4) 
plt.scatter(B.Smokes, B.Alkohol, color="Cyan", label="0", alpha=0.4) 
plt.xlabel(" Smokes ")
plt.ylabel(" Alcohol ")
plt.legend()
plt.title(" Smokes vs Alcohol ")
plt.show()

# spliting up the dataset
x = dataset.iloc[:, 3:5]
y = dataset.iloc[:,6]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# Transforming the values on the same scale
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train) 
x_test = sc_x.transform(x_test)

print('\n---------------**** KNN Algorithm ****---------------') 

# defining a model - KNN

classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')

# fitting the model

classifier.fit(x_train, y_train)

# Predict the output of the test set 

y_pred = classifier.predict(x_test) 
print(y_pred)

# confusion matrix
cm = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix: ")
print(cm)
print("\nIn Confusion Matrix:-----")
print("Position 1.1 shows the patients that don't have Cancer, In this case = 2")
print("Position 1.2 shows the number of patients that have higher risk of Cancer, In this case = 0") 
print("Position 2.1 shows the Incorrect Value, In this case = 0")
print("Position 2.2 shows the correct number of patients that have Cancer, In this case = 1")
