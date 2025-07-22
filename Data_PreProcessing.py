import pandas as pd

def Processed_Data_Train():

    Data_names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    data = pd.read_csv("train.csv", skiprows=1, names=Data_names)
    cleaned_train_data = data.dropna(subset=['Age'])
    cleaned_train_data = cleaned_train_data.drop(['Cabin','Name','Ticket'],axis=1)
    cleaned_train_data['Sex'] = cleaned_train_data['Sex'].map({'male': 1, 'female': 0})
    cleaned_train_data['Embarked'] = cleaned_train_data['Embarked'].map({'Q' : 0, 'C' : 1, 'S' : 2})

    return cleaned_train_data

def Processed_Data_test():

    Data_names = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
    data = pd.read_csv("test.csv", skiprows=1, names=Data_names)
    cleaned_test_data = data.drop(['Cabin','Name','Ticket'],axis=1)
    cleaned_test_data['Sex'] = cleaned_test_data['Sex'].map({'male': 1, 'female': 0})
    cleaned_test_data['Embarked'] = cleaned_test_data['Embarked'].map({'Q' : 0, 'C' : 1, 'S' : 2})

    return cleaned_test_data


