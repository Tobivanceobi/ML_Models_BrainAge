import sys

from skopt.space import Integer, Categorical, Real

sys.path.insert(0, '/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge')
import pandas as pd
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from config import SET_PATH, BASE_PATH
from helper import load_object


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
            max_iter=400,
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

        y_skf = [int(age) for age in data['y']]
        skf_vals = []
        skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
        for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
            skf_vals.append((train_index, test_index))

        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)

        parameter_space = {
            'layer1': Integer(5000, 10000),
            'layer2': Integer(2000, 5000),
            'layer3': Integer(1000, 2000),
            'layer4': Integer(300, 1000),
            'num_hl': Integer(1, 2),
            'activation': Categorical(['tanh', 'relu']),
            'solver': Categorical(['adam']),
            'alpha': Real(0.0001, 0.05),
            'learning_rate': Categorical(['constant', 'adaptive']),
            'learning_rate_init': Real(0.0001, 0.1)
        }

        clf = BayesSearchCV(
            estimator=MLPWrapper(),
            search_spaces=parameter_space,
            n_iter=30,
            cv=skf_vals,
            scoring='neg_mean_absolute_error',
            n_jobs=30
        )

        clf.fit(x, y=y)

        print(clf.cv_results_)
        print(clf.best_score_)
        print(clf.best_params_)
        results = pd.DataFrame(clf.cv_results_)
        f_name = ts.replace('/', '_') + sv.replace('/', '_')
        results.to_csv(BASE_PATH + f'MLP/{f_name}results.csv')
