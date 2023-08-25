import sys

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')

import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV

from config import SET_PATH, BASE_PATH
from helper import load_object

training_sets = ['TS4/']
set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
for ts in training_sets:
    for sv in set_vary:
        set_path = SET_PATH + ts + sv
        data = load_object(set_path + 'training_set')
        x = data['x']
        groups = data['group']
        y = data['y']

        y_skf = [int(age) for age in data['y']]
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
            skf_vals.append((train_index, test_index))

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        param_space = {
            'estimator__n_neighbors': [3, 50],
            'estimator__leaf_size': [3, 300],
            'estimator__p': [1, 2],   # KNN parameter
            'max_samples': (0.1, 1.0),  # Bagging parameter
            'max_features': (0.1, 1.0),  # Bagging parameter
        }

        # Create a KNN Regressor
        knn_regressor = KNeighborsRegressor()

        # Create a Bagging KNN Regressor
        model = BaggingRegressor(estimator=knn_regressor, n_estimators=500, random_state=42, n_jobs=-2)

        clf = BayesSearchCV(estimator=model,
                            search_spaces=param_space,
                            cv=skf_vals,
                            scoring='neg_mean_absolute_error',
                            verbose=4)

        clf.fit(x, y=y)

        print(clf.cv_results_)
        print(clf.best_score_)
        print(clf.best_params_)
        results = pd.DataFrame(clf.cv_results_)
        f_name = ts.replace('/', '_') + sv.replace('/', '_')
        results.to_csv(BASE_PATH + f'BaggedKNN/{f_name}results.csv')
