import sys

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
from config import SET_PATH, BASE_PATH
from helper import load_object, equalize_classes
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

        y_skf = [int(age * 10) for age in data['y']]
        y_skf = equalize_classes(y_skf)
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
            skf_vals.append((train_index, test_index))

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        parameter_space = {
            'iterations': [5000],
            'depth': [2, 6],
            'l2_leaf_reg': [0, 20],
            'learning_rate': [0.0005, 0.2],
            'min_child_samples': [1, 10],
            'random_strength': [0.1, 5],
            'bagging_temperature': [0, 10],
        }

        fit_param = {
            'early_stopping_rounds': 200,
        }

        model = CatBoostRegressor(
            random_seed=126,
            task_type="GPU",
            verbose=False
        )

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
        results.to_csv(BASE_PATH + f'CatBoost/{f_name}results.csv')
