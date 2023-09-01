import sys

from sklearn.linear_model import ElasticNet

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import BASE_PATH, SET_PATH
from helper import load_object, save_object


training_sets = ['TS2/', 'TS4/']
set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
best_ts = ''
best_sv = ''
best_score = -50
for ts in training_sets:
    for sv in set_vary:
        f_name = ts.replace('/', '_') + sv.replace('/', '_')
        res_df = pd.read_csv(BASE_PATH + f'EleasticNet/{f_name}results.csv')
        res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
        best_params = res_df.iloc[0]
        if best_params['mean_test_score'] > best_score:
            best_score = best_params['mean_test_score']
            best_sv = sv
            best_ts = ts

print(best_sv, best_ts, best_score)
set_path = SET_PATH + best_ts + best_sv
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

f_name = best_ts.replace('/', '_') + best_sv.replace('/', '_')
res_df = pd.read_csv(BASE_PATH + f'EleasticNet/{f_name}results.csv')
res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
best_params = res_df.iloc[0]
model_param = dict()
for col in res_df.columns:
    if 'param_' in col:
        key_n = col.replace('param_', '')
        model_param[key_n] = best_params[col]

results = dict()
results['ts'] = best_ts
results['sv'] = best_sv
best_model = ElasticNet()
best_mae = 50
fold_score_mae = []
fold_score_r2 = []
for fold in range(len(skf_vals)):
    x_train = [x[i] for i in skf_vals[fold][0]]
    x_test = [x[i] for i in skf_vals[fold][1]]
    y_train = [y[i] for i in skf_vals[fold][0]]
    y_test = [y[i] for i in skf_vals[fold][1]]

    model = ElasticNet(**model_param, random_state=42, max_iter=12000)
    model.fit(x_train, y=y_train)

    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    fold_score_mae.append(mae)
    fold_score_r2.append(r2)

results['fold_mae'] = fold_score_mae
results['fold_r2'] = fold_score_r2
save_object(results, BASE_PATH + 'EleasticNet/best_model')
