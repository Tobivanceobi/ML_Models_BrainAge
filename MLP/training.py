import sys

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')
import pandas as pd
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
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

        y_skf = [int(age * 10) for age in data['y']]
        y_skf = equalize_classes(y_skf)
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
            skf_vals.append((train_index, test_index))

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        parameter_space = {
            'hidden_layer_sizes': [
                (len(x[0]), int(len(x[0]) / 2)),
                (len(x[0]), int(len(x[0]) / 2), int(len(x[0]) / 4)),
                (len(x[0]), int(len(x[0]) / 4), int(len(x[0]) / 8)),
                (len(x[0]), int(len(x[0]) / 2), int(len(x[0]) / 4), int(len(x[0]) / 8)),
                (len(x[0]), int(len(x[0]) / 4), int(len(x[0]) / 8), int(len(x[0]) / 16))
            ],
            'activation': ['tanh', 'relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.0001, 0.1]
        }

        model = MLPRegressor(max_iter=500)

        clf = BayesSearchCV(estimator=model,
                            search_spaces=parameter_space,
                            cv=skf_vals,
                            n_jobs=50,
                            scoring='neg_mean_absolute_error',
                            verbose=4)

        clf.fit(x, y=y)

        print(clf.cv_results_)
        print(clf.best_score_)
        print(clf.best_params_)
        results = pd.DataFrame(clf.cv_results_)
        results.to_csv('MLP/results.csv')
