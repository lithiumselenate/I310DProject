import pandas as pd

data = pd.read_csv('train_data.csv')
train_features = data[['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']]
train_labels = data["Churn"]
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs',max_iter=10000)
classifier.fit(train_features.to_numpy(),train_labels.to_numpy())
from sklearn.metrics import accuracy_score
test = pd.read_csv("test_data.csv")
test_inputs = test[['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']]
y_actual = test["Churn"]
y_predicted_lr = classifier.predict(test_inputs.to_numpy())
accuracy_score = accuracy_score(y_predicted_lr,y_actual)
print (f"Accuracy of the Logistic Classifier = {accuracy_score}")