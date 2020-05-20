import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Lato-Regular']

train_data = pd.read_csv('train.csv')

"""Drop features that can't predict anything like name, ticket,
and cabin."""
train_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

"""Clean up data by filling in values"""
features = ['Pclass', 'Sex', 'Embarked', 'Age', 'SibSp', 'Parch', 'Fare']
for fname in features:
    if train_data[fname].isna().any():
        """For numerics, fill in with mean. Else just use the most frequent value"""
        if train_data[fname].dtype == 'int64' or train_data[fname].dtype == 'float64':
            train_data[fname].fillna(train_data[fname].mean(), inplace=True)
        elif train_data[fname].dtype == 'object':
            train_data[fname].fillna(train_data[fname].mode()[0], inplace=True)

def plot_total_survival(train_data):
    """Plotting Total Survival Rate"""
    plt.figure(figsize=(7, 5), dpi=120)
    plt.title("Total survival rate of passengers aboard RMS Titanic")
    sns.countplot('Survived', data=train_data)
    plt.show()

plot_total_survival(train_data)

