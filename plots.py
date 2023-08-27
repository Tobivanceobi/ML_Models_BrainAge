import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

from config import freq_bands
from helper import load_object

MODEL_LIST = [
    'BaggedKNN', 'CatBoost', 'EleasticNet',
    'KNN', 'LassoRegression', 'LogisticRegression',
    'MLP', 'RandomForrest', 'SVRegression', 'XGBoost'
]

SET_PATH = r'/home/tobias/Schreibtisch/EEG-FeatureExtraction/trainingSets/TSFinal/'

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

    y_skf = [int(age) for age in data['y']]
    skf_vals = []
    skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=126)
    for fold, (train_index, test_index) in enumerate(skf.split(x, y_skf, groups)):
        skf_vals.append((train_index, test_index))

    x_train = [x[i] for i in skf_vals[fold][0]]
    x_test = [x[i] for i in skf_vals[fold][1]]
    y_train = [y[i] for i in skf_vals[fold][0]]
    y_test = [y[i] for i in skf_vals[fold][1]]

    x_train_df = pd.DataFrame(x_train, columns=x_names)
    x_test_df = pd.DataFrame(x_test, columns=x_names)
    # Group features for aggregation
    feature_groups_fb = []
    n_labels_fb = []
    for fb in freq_bands:
        feature_group_idx = []
        for i in range(len(x_names)):
            if fb in x_names[i]:
                if fb == 'whole_spec':
                    temp = False
                    for ofb in ['delta', 'theta', 'alpha', 'beta']:
                        if ofb in x_names[i]:
                            temp = True
                    if temp:
                        continue
                    else:
                        feature_group_idx.append(i)
                else:
                    feature_group_idx.append(i)
        if len(feature_group_idx) > 0:
            feature_groups_fb.append(feature_group_idx)
            n_labels_fb.append(fb)

    # Calculate aggregated SHAP values for each feature group
    grouped_shap_values = np.zeros((len(x_test), len(feature_groups_fb)))
    for i, group in enumerate(feature_groups_fb):
        grouped_shap_values[:, i] = np.sum(shap_values[:, group], axis=1)

    shap.initjs()
    shap.summary_plot(grouped_shap_values, feature_names=n_labels_fb)
    # shap.summary_plot(shap_values, x_test_df, max_display=100)
    plt.show()
    vals = np.abs(grouped_shap_values).mean(0)
    vals, n_labels = zip(*sorted(zip(vals, n_labels_fb), reverse=True))
    print(vals)
    print(n_labels)
