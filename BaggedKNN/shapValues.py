import sys

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from config import SET_PATH, BASE_PATH
from helper import load_object, save_object


training_sets = ['TS2/']
set_vary = ['meanEpochs/']
for ts in training_sets:
    for sv in set_vary:
        set_path = SET_PATH + ts + sv
        data = load_object(set_path + 'training_set')

        x = data['x']
        groups = data['group']
        y = data['y']
        x_names = data['x_names']

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        y_skf = [int(age) for age in data['y']]
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
            skf_vals.append((train_index, test_index))

        f_name = ts.replace('/', '_') + sv.replace('/', '_')
        res_df = pd.read_csv(BASE_PATH + f'BaggedKNN/{f_name}results.csv')
        res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
        best_params = res_df.iloc[0]

        model_param = dict()
        for col in res_df.columns:
            if 'param_' in col and not('base_estimator__' in col):
                key_n = col.replace('param_', '')
                model_param[key_n] = best_params[col]

        best_fold = 0
        best_score = 5
        best_model = ElasticNet(random_state=42, max_iter=5000)
        for fold in range(len(skf_vals)):
            x_train = [x[i] for i in skf_vals[fold][0]]
            x_test = [x[i] for i in skf_vals[fold][1]]
            y_train = [y[i] for i in skf_vals[fold][0]]
            y_test = [y[i] for i in skf_vals[fold][1]]

            # Create a KNN Regressor
            knn_regressor = KNeighborsRegressor()

            # Create a Bagging KNN Regressor
            model = BaggingRegressor(**model_param, base_estimator=knn_regressor, random_state=42, n_jobs=-2)
            model.fit(x_train, y=y_train)

            preds = model.predict(x_test)
            mae = mean_absolute_error(y_test, preds)
            if mae < best_score:
                best_fold = fold
                best_score = mae
                best_model = model

        print(best_score)

        x_train = [x[i] for i in skf_vals[best_fold][0]]
        x_test = [x[i] for i in skf_vals[best_fold][1]]
        y_train = [y[i] for i in skf_vals[best_fold][0]]
        y_test = [y[i] for i in skf_vals[best_fold][1]]

        x_train_df = pd.DataFrame(x_train, columns=x_names)
        x_test_df = pd.DataFrame(x_test, columns=x_names)

        # Initialize the shap explainer
        explainer = shap.KernelExplainer(best_model.predict, shap.sample(x_train_df, 10), num_jobs=-2)

        # Compute Shap values for all instances in X_test
        shap_values = explainer.shap_values(x_test_df)

        shap_dict = dict(
            shap_values=shap_values,
            fold=skf_vals[best_fold]
        )

        save_object(shap_dict, BASE_PATH + f'BaggedKNN/shap_values')


