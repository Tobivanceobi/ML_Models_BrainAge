import sys

import config

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from config import SET_PATH
from helper import load_object, equalize_classes
import pandas as pd
from skopt import BayesSearchCV

training_sets = ['TS2/']
set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
for ts in training_sets:
    for sv in set_vary:
        set_path = SET_PATH + ts + sv
        data = load_object(set_path + 'training_set')
        x = data['x']
        groups = data['group']
        y = data['y']

        y = [int(age * 10) for age in data['y']]
        y = equalize_classes(y)
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y, groups)):
            skf_vals.append((train_index, test_index))

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        model = LogisticRegression(n_jobs=60)

        # Define the parameter search space for Logistic Regression
        parameter_space = {
            'C': (1e-6, 1e+6, 'log-uniform'),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

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
        results.to_csv(config.BASE_PATH + f'LogisticRegression/{f_name}results.csv')
