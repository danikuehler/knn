#Danielle Kuehler
#ITP 449 Summer 2020
#HW8
#Q1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#1. Import file to dataframe "diabetes_knn" and display all columns
diabetes_knn = pd.read_csv("diabetes.csv")
pd.set_option("Display.max_columns", None) #Display all columns
print(diabetes_knn) #Display dataframe

#2. Dimensions of the dataframe
print("Dimensions: ", diabetes_knn.shape) #Shape shows rows,columns

#3. Check for null values- There are none
print("\n4. There are no missing values\n", diabetes_knn.isnull().any()) #There are no missing values

#4. Feature matrix (X) and target vector (y)
X = diabetes_knn.iloc[:,0:8] #Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
y = diabetes_knn.iloc[:,8] #Column: Outcome

#5. Standardize the attributes of feature matrix
scaler = StandardScaler()
scaler.fit(X) #Fit it to feature matrix
diabetes_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns) #Use transform function of standardizer, convert fitted data into dataframe

#6. Split the Feature Matrix and Target Vector into training and testing sets, reserving 30% of the data for testing. random_state = 10, stratify = y
X_train, X_test, y_train, y_test = \
    train_test_split(diabetes_scaled, y, test_size=0.3, random_state=10, stratify=y)

#7.Develop a KNN based model and obtain KNN score (accuracy) for train and test data for k’s values ranging between 1 to 15
neighbors = np.arange(1,16)
train_accuracy = np.empty(15)
test_accuracy = np.empty(15)

for k in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=k) #Iterate in range to find best k
    knn.fit(X_train, y_train) #Fit data
    y_pred = knn.predict(X_test) #Predict value
    cf = metrics.confusion_matrix(y_test, y_pred) #Confusion matrix
    train_accuracy[k-1] = knn.score(X_train, y_train) #Store accuracy of training variables in list
    test_accuracy[k-1] = knn.score(X_test, y_test) #Store accuracy of testing variables in list

#8.	Plot a graph of train and test score and determine the best value of k.
plt.figure(2) #Create figure
plt.title("KNN: Varying Number of Neighbors") #Title
plt.plot(neighbors, test_accuracy, label = "Testing Accuracy") #Plot testing accuracy
plt.plot(neighbors, train_accuracy, label = "Training Accuracy") #Plot traning accuracy
plt.legend() #Display legend of each line color
#Formatting
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show() #Display
#Best value of k is 15

#9.	Display the test score of the model with best value of k and print the confusion matrix for it.
#Instantiate KNN again with k = 15
X_train, X_test, y_train, y_test = \
    train_test_split(diabetes_scaled, y, test_size=0.3, random_state=10, stratify=y)
knn = KNeighborsClassifier(n_neighbors = 15)  #K = 15
knn.fit(X_train, y_train) #Fit data
y_pred = knn.predict(X_test) #Predict values
cf = metrics.confusion_matrix(y_test, y_pred) #Create confusion matrix
test_accuracy = knn.score(X_test, y_test) #Calculate accuracy

print("Test score with k = 15:", test_accuracy) #Testing score
labels = ["Yes","No"]
cf_df = pd.DataFrame(cf, index = labels, columns = labels)
print("Confusion matrix at k = 15:\n", cf_df) #Confusion matrix

#10. Predict the Outcome for a person with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age.
newFrame = pd.DataFrame([[2,150,85,22,200,30,.3,55]]) #Numbers correspond to attributes of person
survive_Predict = knn.predict(newFrame)
if survive_Predict:
    print("A person with with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age will have diabetes.")
else:
    print("A person with with 2 pregnancies, 150 glucose, 85 blood pressure, 22 skin thickness, 200 insulin, 30 BMI, 0.3 diabetes pedigree, 55 age will not have diabetes.")

