import sys

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from config import BASE_PATH, SET_PATH
from helper import load_object, save_object


class MLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 layer1=None,
                 layer2=None,
                 num_hl=None,
                 batch_size=None,
                 activation=None,
                 solver=None,
                 learning_rate=None,
                 learning_rate_init=None,
                 alpha=None):
        self.layer1 = layer1
        self.layer2 = layer2
        self.num_hl = num_hl
        self.batch_size = batch_size
        self.activation = activation
        self.solver = solver
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha

    def fit(self, x_train, y_train):
        print([self.layer1, self.layer2][-1 * self.num_hl:])
        print(self.solver)
        print(self.learning_rate_init)
        print(self.activation)
        print(self.alpha)
        model_mlp = MLPRegressor(
            hidden_layer_sizes=[self.layer1, self.layer2][-1 * self.num_hl:],
            max_iter=1000,
            activation=self.activation,
            batch_size=self.batch_size,
            solver=self.solver,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha
        )
        model_mlp.fit(x_train, y_train)
        self.model = model_mlp
        return self

    def predict(self, x_train):
        return self.model.predict(x_train)

    def score(self, x_train, y_train):
        return self.model.score(x_train, y_train)


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
        res_df = pd.read_csv(BASE_PATH + f'MLP/{f_name}results.csv')
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

            model = MLPWrapper(**model_param)
            model.fit(x_train, y_train)

            preds = model.predict(x_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            fold_score_mae.append(mae)
            fold_score_r2.append(r2)
        res_dict['fold_mae'] = fold_score_mae
        res_dict['fold_r2'] = fold_score_r2
        results[ts][sv] = res_dict

save_object(results, BASE_PATH + 'MLP/best_model')
