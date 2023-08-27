import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

from config import freq_bands
from helper import load_object, group_freq_bands_shap

MODEL_LIST = [
    'BaggedKNN', 'EleasticNet',
    'KNN', 'LassoRegression', 'LogisticRegression',
    'MLP', 'RandomForrest', 'SVRegression'
]

SET_PATH = r'/home/tobias/Schreibtisch/EEG-FeatureExtraction/trainingSets/TSFinal/'

result_df = dict(
    model=[],
    feature_rank=[],
    shap_vals=[]
)

for m in MODEL_LIST:
    set_path = SET_PATH + 'TS2/meanEpochs/'
    data = load_object(set_path + 'training_set')
    x = data['x']
    groups = data['group']
    y = data['y']
    x_names = data['x_names']

    shap_dict = load_object(m + '/' + 'shap_values')
    fold = shap_dict['fold']
    shap_values = shap_dict['shap_values']

    x_train = [x[i] for i in fold[0]]
    x_test = [x[i] for i in fold[1]]
    y_train = [y[i] for i in fold[0]]
    y_test = [y[i] for i in fold[1]]

    x_train_df = pd.DataFrame(x_train, columns=x_names)
    x_test_df = pd.DataFrame(x_test, columns=x_names)

    # Group features for aggregation
    n_labels_fb, feature_groups_fb = group_freq_bands_shap(x_names)

    # Calculate aggregated SHAP values for each feature group
    grouped_shap_values = np.zeros((len(x_test), len(feature_groups_fb)))
    for i, group in enumerate(feature_groups_fb):
        grouped_shap_values[:, i] = np.sum(shap_values[:, group], axis=1)

    vals = np.abs(grouped_shap_values).mean(0)
    vals, n_labels = zip(*sorted(zip(vals, n_labels_fb), reverse=True))

    result_df['feature_rank'].append(n_labels)
    result_df['model'].append(m)
    result_df['shap_vals'].append(vals)

for m in range(len(result_df['model'])):

    rank_order = sorted(result_df['feature_rank'][0], key=lambda elem: sum(result_df['feature_rank']) / len(result_df['feature_rank']))
    print(rank_order)
