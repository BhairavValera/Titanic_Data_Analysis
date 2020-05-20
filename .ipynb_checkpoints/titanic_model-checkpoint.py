import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

def missing_values_fill(features, X):
    for fname in features:
        if X[fname].isna().any():
            """For numerics, fill in with mean. Else just use the most frequent value"""
            if X[fname].dtype == 'int64' or X[fname].dtype == 'float64':
                X[fname].fillna(X[fname].mean(), inplace=True)
            elif X[fname].dtype == 'object':
                X[fname].fillna(X[fname].mode()[0], inplace=True)

X_full = pd.read_csv('train.csv')
X_test_full = pd.read_csv('test.csv')

y = X_full['Survived']

"""We separate the target from the predictors."""
X_full.drop(['Survived'], axis=1, inplace=True)

"""Drop features that can't predict anything like name, ticket,
and cabin."""
X_full.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

"""For all features that have missing values, fill them in accordingly"""
features = ['Pclass', 'Sex', 'Embarked', 'Age', 'SibSp', 'Parch', 'Fare']
missing_values_fill(features, X_full)

"""Generate training and validation data"""
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)

"""Selecting the featured training data from split to build model"""
X_train = X_train_full[features].copy()
X_valid = X_valid_full[features].copy()

"""One-hot-encode the categorical values"""
categorical_features = [fname for fname in X_full.columns
                        if X_full[fname].dtype == 'object']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_X_train = pd.DataFrame(OH_encoder.fit_transform(X_train[categorical_features]))
OH_X_valid = pd.DataFrame(OH_encoder.transform(X_valid[categorical_features]))

OH_X_train.index = X_train.index
OH_X_valid.index = X_valid.index

numerical_X_train = X_train.drop(categorical_features, axis=1)
numerical_X_valid = X_valid.drop(categorical_features, axis=1)

OH_X_train = pd.concat([numerical_X_train, OH_X_train], axis=1)
OH_X_valid = pd.concat([numerical_X_valid, OH_X_valid], axis=1)

"Specifying a model"
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

"""Fitting"""
rf_model.fit(OH_X_train, y_train)

"""Testing model"""
predictions = rf_model.predict(OH_X_valid)
print(rf_model.score(OH_X_valid, y_valid))

"""Predicting with test data. Again, drop features that can't predict anything like name, ticket,
and cabin."""
X_test_full.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

"""For all features that have missing values, fill them in accordingly"""
missing_values_fill(features, X_test_full)
X_test = X_test_full[features].copy()

"""One-hot-encode the categorical values"""
OH_X_test = pd.DataFrame(OH_encoder.fit_transform(X_test[categorical_features]))
OH_X_test.index = X_test.index
numerical_X_test = X_test.drop(categorical_features, axis=1)
OH_X_test = pd.concat([numerical_X_test, OH_X_test], axis=1)

"""Make predictions with test data"""
test_predictions = rf_model.predict(OH_X_test)
output = pd.DataFrame({'PassengerId': X_test_full.PassengerId,
                       'Survived': test_predictions})
output.to_csv('submission.csv', index=False)