import numpy as np

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

    shap_values = load_object(m + '/' + 'shap_values')
    print(shap_values.shape)
    # Group features for aggregation
    feature_groups_fb = []
    n_labels_fb = []
    for fb in freq_bands:
        feature_group_idx = []
        for i in range(len(x_names)):
            if fb in x_names[i]:
                if fb == 'whole_spec':
                    temp = False
                    for ofb in freq_bands.remove(fb):
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
    grouped_shap_values = np.zeros((len(x_test_df), len(feature_groups)))
    for i, group in enumerate(feature_groups):
        grouped_shap_values[:, i] = np.sum(shap_values[:, group], axis=1)

    shap.initjs()
    shap.summary_plot(grouped_shap_values, feature_names=n_labels)
    # shap.summary_plot(shap_values, x_test_df, max_display=100)
    plt.show()
    vals = np.abs(grouped_shap_values).mean(0)
    vals, n_labels = zip(*sorted(zip(vals, n_labels), reverse=True))
    print(vals)
    print(n_labels)
