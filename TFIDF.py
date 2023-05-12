from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from operator import itemgetter
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from Get_features import clean_text
from sklearn.metrics import classification_report

# Load the data (tweet)
data = pd.read_csv("image_text (copy).csv")
data['clean'] = data['Text'].apply(clean_text)

X = data['clean']
Y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

# Pipeline parameters
parameters = [
    {
        'clf': [SVC()],
        'vect__stop_words': ['english', None],
        'vect__max_df': (0.25, 0.5, 0.75, 1.0),
        'vect__max_features': (None, 50, 100, 250, 500, 1000),
        'clf__C': [0.001, 0.1, 1, 10, 100, 10e5],
        'clf__kernel': ['linear', 'rbf'],
        'clf__class_weight': ['balanced'],
        'clf__probability': [True]
    },
    {
        'clf': [LinearSVC()],
        'vect__stop_words': ['english', None],
        'vect__max_df': (0.25, 0.5, 0.75, 1.0),
        'vect__max_features': (None, 50, 100, 250, 500, 1000),
        'tfidf__use_idf': (True, False),
        'clf__C': [0.001, 0.1, 1, 10, 100, 10e5],
        'clf__class_weight': ['balanced']
    },
    {
        'clf': [DecisionTreeClassifier()],
        'vect__stop_words': ['english', None],
        'vect__max_df': (0.25, 0.5, 0.75, 1.0),
        'vect__max_features': (None, 50, 100, 250, 500, 1000),
        'tfidf__use_idf': (True, False),
        'clf__criterion': ['gini', 'entropy'],
        'clf__splitter': ['best', 'random'],
        'clf__class_weight': ['balanced', None]
    },
    {
        'clf': [LogisticRegression()],
        'vect__stop_words': ['english', None],
        'vect__max_df': (0.25, 0.5, 0.75, 1.0),
        'vect__max_features': (None, 50, 100, 250, 500, 1000),
        'tfidf__use_idf': (True, False),
        'clf__penalty': ['l1', 'l2'],
        'clf__C': np.logspace(-4, 4, 20),
        'clf__solver': ['liblinear'],
        'clf__dual': [False]
    },
    {
        'clf': [GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8,
                                           random_state=10)],
        'vect__stop_words': ['english', None],
        'vect__max_df': (0.25, 0.5, 0.75, 1.0),
        'vect__max_features': (None, 50, 100, 250, 500, 1000),
        'tfidf__use_idf': (True, False),
        'clf__max_depth': range(5, 16, 2),
        'clf__min_samples_split': range(200, 1001, 200)
    }
]

result = []

for params in parameters:
    # Classifier
    clf = params['clf'][0]

    # Getting arguments by popping out classifier
    params.pop('clf')

    # Pipeline
    steps = [
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', clf),
    ]

    # Cross-validation using Grid Search
    grid = GridSearchCV(Pipeline(steps), param_grid=params, cv=3)
    grid.fit(X_train, y_train)

    # Storing result
    result.append(
        {
            'grid': grid,
            'classifier': grid.best_estimator_,
            'best score': grid.best_score_,
            'best params': grid.best_params_,
            'cv': grid.cv
        }
    )

# Sorting result by best score
result = sorted(result, key=itemgetter('best score'), reverse=True)

# Print result
for res in result:
    print(f"Best Score: {res['best score']}")
    print(f"Best Params: {res['best params']}")
    print()

# Saving best classifier
import joblib

grid = result[0]['grid']
joblib.dump(grid, 'clf_Tfidf_img.pickle')
