import sys

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV

from config import SET_PATH
from helper import load_object, equalize_classes


training_sets = ['TS2/']
set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
for ts in training_sets:
    for sv in set_vary:
        set_path = SET_PATH + ts + sv
        data = load_object(set_path + 'training_set')
        x = data['x']
        groups = data['group']
        y = data['y']

        y_skf = [int(age*10) for age in data['y']]
        y_skf = equalize_classes(y_skf)
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y, groups)):
            skf_vals.append((train_index, test_index))

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        parameter_space = {
            'alpha': (1e-5, 1e2, 'log-uniform'),  # Range of alpha values (log-uniformly distributed)
            'l1_ratio': (0.0, 1.0),  # Range of l1_ratio values (between 0 and 1)
        }

        # Create an Elastic Net Regression model
        model = ElasticNet(random_state=42)

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
        results.to_csv('EleasticNet/results.csv')
