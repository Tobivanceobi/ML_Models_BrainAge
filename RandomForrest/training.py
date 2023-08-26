import sys

from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')
from sklearn.model_selection import StratifiedGroupKFold
from config import BASE_PATH, SET_PATH
from helper import load_object
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV

training_sets = ['TS4/']
set_vary = ['meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
for ts in training_sets:
    for sv in set_vary:
        set_path = SET_PATH + ts + sv
        data = load_object(set_path + 'training_set')
        x = data['x']
        groups = data['group']
        y_org = data['y']
        y = [int(age * 10) for age in data['y']]

        le = LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        y_skf = [int(age) for age in data['y']]
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
            skf_vals.append((train_index, test_index))

        parameter_space = {
            'max_depth': [45, 100],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [2, 9],
            'min_weight_fraction_leaf': [0, 0.5],
            'min_impurity_decrease': [0, 0.9],
            'n_estimators': [300, 1000]
        }

        model = RandomForestRegressor(n_jobs=-2)

        fit_param = {
            'early_stopping_rounds': 200,
        }

        clf = BayesSearchCV(estimator=model,
                            search_spaces=parameter_space,
                            fit_params=fit_param,
                            cv=skf_vals,
                            scoring='neg_mean_absolute_error',
                            verbose=4)

        clf.fit(x, y=y)

        print(clf.cv_results_)
        print(clf.best_score_)
        print(clf.best_params_)
        results = pd.DataFrame(clf.cv_results_)
        f_name = ts.replace('/', '_') + sv.replace('/', '_')
        results.to_csv(BASE_PATH + f'RandomForrest/{f_name}results.csv')
