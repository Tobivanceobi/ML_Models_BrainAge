import sys

from sklearn.svm import SVR

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import BASE_PATH, SET_PATH
from helper import load_object, save_object

training_sets = ['TS2/', 'TS4/']
set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
results = dict()
for ts in training_sets:
    results[ts] = dict()
    for sv in set_vary:
        set_path = SET_PATH + ts + sv

        data = load_object(set_path + 'training_set')
        x = data['x']
        groups = data['group']
        y = data['y']
        x_names = data['x_names']

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        y_skf = [int(age) for age in data['y']]
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
            skf_vals.append((train_index, test_index))

        f_name = ts.replace('/', '_') + sv.replace('/', '_')
        res_df = pd.read_csv(BASE_PATH + f'SVRegression/{f_name}results.csv')
        res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
        best_params = res_df.iloc[0]
        model_param = dict()
        for col in res_df.columns:
            if 'param_' in col:
                key_n = col.replace('param_', '')
                model_param[key_n] = best_params[col]

        res_dict = dict()
        fold_score_mae = []
        fold_score_r2 = []
        for fold in range(len(skf_vals)):
            x_train = [x[i] for i in skf_vals[fold][0]]
            x_test = [x[i] for i in skf_vals[fold][1]]
            y_train = [y[i] for i in skf_vals[fold][0]]
            y_test = [y[i] for i in skf_vals[fold][1]]

            model = SVR(**model_param, max_iter=6000)
            model.fit(x_train, y=y_train)

            preds = model.predict(x_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            fold_score_mae.append(mae)
            fold_score_r2.append(r2)
        res_dict['fold_mae'] = fold_score_mae
        res_dict['fold_r2'] = fold_score_r2
        results[ts][sv] = res_dict

save_object(results, BASE_PATH + 'SVRegression/best_model')
