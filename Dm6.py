!pip install google-auth-oauthlib


# Step 1: Import libraries
! pip install -U -q PyDrive
from  pydrive.auth  import  GoogleAuth
from  pydrive.drive  import  GoogleDrive
from  google.colab  import  auth
from  oauth2client.client  import  GoogleCredentials

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Autheticate E-Mail ID
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Step 2: Load the dataset

#https://drive.google.com/file/d/1vJ-ttW1INyqvA771el_hVQz8dbP55zgz/view?usp=sharing (Dataset Downl oads Link)

# Get File from Drive using file-ID
downloaded = drive.CreateFile ({ 'id' : '1vJ-ttW1INyqvA771el_hVQz8dbP55zgz' }) # replace the id with id of file you want to access s
downloaded.GetContentFile ( 'iris.csv' )
iris_data = pd.read_csv ( 'iris.csv' )
iris_data

# Step 3: Data preprocessing
# Split the dataset into features (X) and target variable (y)
X = iris_data.drop(columns=['variety'])
y = iris_data['variety']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose Classification Techniques
# Initialize classifiers for Decision Trees, SVM, and KNN
dt_classifier = DecisionTreeClassifier()
svm_classifier = SVC()
knn_classifier = KNeighborsClassifier()

# Step 5: Train and Evaluate Models
# Train each model using the training data
dt_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)

# Make predictions on the testing data
dt_predictions = dt_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)

# Calculate accuracy for each model using accuracy_score()
dt_accuracy = accuracy_score(y_test, dt_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Step 6: Compare Results
print("Decision Tree Accuracy:", dt_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)