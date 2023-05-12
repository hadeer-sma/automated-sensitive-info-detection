from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from operator import itemgetter
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.naive_bayes import GaussianNB
import joblib

# Load the data
data = pd.read_csv("../test_with_features.csv")
X = data[['Person', 'Loc', 'Date', 'Gpe', 'Time', 'Place', 'Pos', 'Neg', 'Url', 'Tag']]
Y = data['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

# Define numeric features and their transformers
numeric_features = ['Person', 'Loc', 'Date', 'Gpe', 'Time', 'Place', 'Pos', 'Neg', 'Url', 'Tag']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessor for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Define pipeline parameters for different classifiers
parameters = [
    {
        'clf': [SVC()],
        'clf__C': [0.001, 0.1, 1, 10, 100, 10e5],
        'clf__kernel': ['linear', 'rbf'],
        'clf__class_weight': ['balanced'],
        'clf__probability': [True]
    },
    {
        'clf': [LinearSVC()],
        'clf__C': [0.001, 0.1, 1, 10, 100, 10e5],
        'clf__class_weight': ['balanced'],
        'clf__penalty': ['l1', 'l2'],
        'clf__loss': ['hinge', 'squared_hinge']
    },
    {
        'clf': [DecisionTreeClassifier()],
        'clf__criterion': ['gini', 'entropy'],
        'clf__splitter': ['best', 'random'],
        'clf__class_weight': ['balanced', None]
    },
    {
        'clf': [LogisticRegression()],
        'clf__penalty': ['l1', 'l2'],
        'clf__C': np.logspace(-4, 4, 20),
        'clf__solver': ['liblinear']
    },
    {
        'clf': [GaussianNB()],
        'clf__priors': [None],
        'clf__var_smoothing': [0.00000001, 0.000000001, 0.00000001]
    }
]

result = []

# Iterate over different parameter configurations
for params in parameters:
    # Get the classifier
    clf = params['clf'][0]

    # Remove classifier from parameters
    params.pop('clf')

    # Define the pipeline with preprocessor and classifier
    steps = [('preprocessor', preprocessor), ('clf', clf)]

    # Perform Grid Search with cross-validation
    grid = GridSearchCV(Pipeline(steps), param_grid=params, cv=3)
    grid.fit(X_train, y_train)

    # Store the results
    result.append({
        'grid': grid,
        'classifier': grid.best_estimator_,
        'best score': grid.best_score_,
        'best params': grid.best_params_,
        'cv': grid.cv
    })

    # Sort the results by the best score
    result = sorted(result, key=itemgetter('best score'), reverse=True)

    # Save the best classifier
    grid = result[0]['grid']
    joblib.dump(grid, '../clf_statistic.pickle')
