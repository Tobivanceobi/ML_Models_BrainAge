import sys

from sklearn.base import BaseEstimator, RegressorMixin
from skopt.space import Categorical, Integer, Real

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from config import SET_PATH, BASE_PATH
from helper import load_object
import pandas as pd
from skopt import BayesSearchCV


training_sets = ['TS2/', 'TS4/']
set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
for ts in training_sets:
    for sv in set_vary:
        set_path = SET_PATH + ts + sv
        data = load_object(set_path + 'training_set')
        x = data['x']
        groups = data['group']
        y = [int(age * 10) for age in data['y']]

        y_skf = [int(age) for age in data['y']]
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
            skf_vals.append((train_index, test_index))

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        model = LogisticRegression(max_iter=1000, n_jobs=-2)

        # Define the parameter search space for Logistic Regression
        parameter_space = [
            {
                "C": Integer(1, 1000),
                "solver": Categorical(['liblinear']),
                "penalty": Categorical(['l1', 'l2']),
                "fit_intercept": Categorical([True, False]),
            },
            {
                "C": Integer(1, 1000),
                "solver": Categorical(['lbfgs', 'newton-cg', 'sag']),
                "penalty": Categorical(['l2', 'none']),
                "fit_intercept": Categorical([True, False]),
            },
            {
                "solver": Categorical(['saga']),
                "penalty": Categorical(['l1', 'l2', 'none']),
                "fit_intercept": Categorical([True, False]),
            },
            {
                "C": Integer(1, 1000),
                "solver": Categorical(['saga']),
                "penalty": Categorical(['elasticnet']),
                "fit_intercept": Categorical([True, False]),
                "l1_ratio": Real(0, 1, prior='uniform'),
            },
        ]

        clf = BayesSearchCV(estimator=model,
                            search_spaces=parameter_space,
                            cv=skf_vals,
                            scoring='neg_mean_absolute_error',
                            verbose=4)

        clf.fit(x, y=y)

        print(clf.cv_results_)
        print(clf.best_score_)
        print(clf.best_params_)
        results = pd.DataFrame(clf.cv_results_)
        f_name = ts.replace('/', '_') + sv.replace('/', '_')
        results.to_csv(BASE_PATH + f'LogisticRegression/{f_name}results.csv')
