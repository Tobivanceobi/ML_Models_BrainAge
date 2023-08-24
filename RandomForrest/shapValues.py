import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from config import SET_PATH, BASE_PATH
from helper import load_object, str_to_dict, save_object

methods = [
    'pow_freq_bands',
    'svd_fisher_info',
    'hjorth_complexity_spect',
    'wavelet_coef_energy',
    'hjorth_complexity',
    'spect_slope',
    'std',
    'ptp_amp',
    'quantile',
    'line_length',
    'zero_crossings',
    'skewness',
    'kurtosis',
    'higuchi_fd',
    'samp_entropy',
    'app_entropy',
    'spect_entropy',
    'mean',
    'hurst_exp'
]

freq_bands = ['delta', 'theta', 'alpha', 'beta', 'whole_spec']

shap_dict = dict()
training_sets = ['TS2/', 'TS4/']
set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
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
        res_df = pd.read_csv(BASE_PATH + f'RandomForrest/{f_name}results.csv')

        res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
        model_param = str_to_dict(res_df.iloc[0]['params'])
        print(model_param.keys())

        best_fold = 0
        best_score = 5
        best_model = RandomForestRegressor()
        for fold in range(len(skf_vals)):
            x_train = [x[i] for i in skf_vals[fold][0]]
            x_test = [x[i] for i in skf_vals[fold][1]]
            y_train = [y[i] for i in skf_vals[fold][0]]
            y_test = [y[i] for i in skf_vals[fold][1]]

            model = RandomForestRegressor(**model_param, n_estimators=4000, n_jobs=30)
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
        explainer = shap.Explainer(best_model, x_train_df, num_jobs=30)

        # Compute Shap values for all instances in X_test
        shap_values = explainer(x_test_df)

        # Print Shap values for the first instance
        print("Shap values for the first instance:\n", shap_values[0])

        feature_groups = []
        n_labels = []
        for fb in freq_bands:
            for m in methods:
                n = fb + '_' + m

                fg = [idx_g for idx_g in range(len(x_names)) if n in x_names[idx_g]]
                if len(fg) > 2:
                    n_labels.append(n)
                    feature_groups.append(fg)

        # Calculate aggregated SHAP values for each feature group
        grouped_shap_values = np.zeros((len(x_test_df), len(feature_groups)))
        for i, group in enumerate(feature_groups):
            grouped_shap_values[:, i] = np.sum(shap_values.values[:, group], axis=1)

        vals = np.abs(grouped_shap_values).mean(0)
        vals, n_labels = zip(*sorted(zip(vals, n_labels), reverse=True))
        print(vals)
        print(n_labels)
        data = np.array([n_labels, vals]).transpose()
        result = pd.DataFrame(data, columns=['Feature Name', 'ShapVals'])
        shap_dict[f_name] = result
        save_object(shap_dict, BASE_PATH + f'RandomForrest/shap_values')


