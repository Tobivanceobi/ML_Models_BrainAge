import sys

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')
import pandas as pd
import shap
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from config import SET_PATH, BASE_PATH
from helper import load_object, save_object


class MLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,
                 layer1=None,
                 layer2=None,
                 layer3=None,
                 layer4=None,
                 num_hl=None,
                 activation=None,
                 solver=None,
                 learning_rate=None,
                 learning_rate_init=None,
                 alpha=None):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.num_hl = num_hl
        self.activation = activation
        self.solver = solver
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha

    def fit(self, x_train, y_train):
        print([self.layer1, self.layer2, self.layer3, self.layer4][-1*self.num_hl:])
        print(self.solver)
        print(self.learning_rate_init)
        print(self.activation)
        print(self.alpha)
        model_mlp = MLPRegressor(
            hidden_layer_sizes=[self.layer1, self.layer2, self.layer3, self.layer4][-1*self.num_hl:],
            max_iter=1000,
            activation=self.activation,
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


training_sets = ['TS2/']
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
        res_df = pd.read_csv(BASE_PATH + f'MLP/{f_name}results.csv')
        res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
        best_params = res_df.iloc[0]

        model_param = dict()
        for col in res_df.columns:
            if 'param_' in col:
                key_n = col.replace('param_', '')
                model_param[key_n] = best_params[col]

        best_fold = 0
        best_score = 5
        best_model = MLPWrapper()
        for fold in range(len(skf_vals)):
            x_train = [x[i] for i in skf_vals[fold][0]]
            x_test = [x[i] for i in skf_vals[fold][1]]
            y_train = [y[i] for i in skf_vals[fold][0]]
            y_test = [y[i] for i in skf_vals[fold][1]]

            model = MLPWrapper(**model_param)
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
        explainer = shap.KernelExplainer(best_model.predict, shap.sample(x_train_df, 10), num_jobs=30)

        # Compute Shap values for all instances in X_test
        shap_values = explainer.shap_values(x_test_df)

        save_object(shap_values, BASE_PATH + f'MLP/shap_values')


