# As I failed to configure local jupyter notebook server's python edition, I am working with this py file.
import pandas as pd
import numpy as np
# Data preprocessing
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.drop('customerID', axis = 1, inplace=True)
#Data preprocessing
def female_male(str1):
    if str1=='Female':
         return 1
    return -1
data['gender']= data['gender'].apply(female_male)

def changeservice_toNum(str1):
    if str1=='Yes':
        return 1
    if str1=='No':
        return 0
    return -1
def InternetService_toNum(str1):
    if (str1=="DSL"):
        return 1
    if (str1=="Fiber optic"):
        return 2
    return 0
# I know that you can do this with 1 line lol, but I am not sure with this.
data['Partner'] = data['Partner'].apply(changeservice_toNum)
data['Dependents'] = data['Dependents'].apply(changeservice_toNum)
data['PhoneService'] = data['PhoneService'].apply(changeservice_toNum)
data['MultipleLines']= data['MultipleLines'].apply(changeservice_toNum)
data['InternetService'] = data['InternetService'].apply(InternetService_toNum)
data['OnlineSecurity']= data['OnlineSecurity'].apply(changeservice_toNum)
data['OnlineBackup'] = data['OnlineBackup'].apply(changeservice_toNum)
data['TechSupport'] = data['TechSupport'].apply(changeservice_toNum)
data['StreamingTV'] = data['StreamingTV'].apply(changeservice_toNum)
data['StreamingMovies'] = data['StreamingMovies'].apply(changeservice_toNum)
data['PaperlessBilling'] = data['PaperlessBilling'].apply(changeservice_toNum)
data['Churn'] = data['Churn'].apply(changeservice_toNum)
data['DeviceProtection'] = data['DeviceProtection'].apply(changeservice_toNum)
def Contract_toNum(str1):
    if str1=='Month-to-month':
        return 1
    if str1=='One year':
        return 2
    if str1=='Two year':
        return 3
    return 0   
data['Contract']= data['Contract'].apply(Contract_toNum)
def Payment_toNum(str1):
    if str1=='Electronic check':
        return 1
    if str1=='Mailed check':
        return 2
    if str1=='Bank transfer (automatic)':
        return 3
    if str1=='Credit card (automatic)':
        return 4
    return 0
data['PaymentMethod'] = data['PaymentMethod'].apply(Payment_toNum)
data.to_csv('Processsed_data.csv')



