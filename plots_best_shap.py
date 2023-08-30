import mne
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from config import chan_map
from helper import load_object
from plot_helper import group_chan_fb, plot_topo_vals_12

MODEL_LIST = [
    'EleasticNet',
    'KNN', 'LassoRegression',
    'MLP', 'SVRegression'
]

SET_PATH = r'/home/tobias/Schreibtisch/EEG-FeatureExtraction/trainingSets/TSFinal/'

result_df = dict(
    model=[],
    feature_rank=[],
    shap_vals=[]
)


set_path = SET_PATH + 'TS2/meanEpochs/'
data = load_object(set_path + 'training_set')
x = data['x']
groups = data['group']
y = data['y']
x_names = data['x_names']
y = [int(age * 10) for age in data['y']]
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

shap_dict = load_object('XGBoost/shap_values')
fold = shap_dict['fold']
shap_values = shap_dict['shap_values']

x_train = [x[i] for i in fold[0]]
x_test = [x[i] for i in fold[1]]
y_train = [y[i] for i in fold[0]]
y_test = [y[i] for i in fold[1]]

x_train_df = pd.DataFrame(x_train, columns=x_names)
x_test_df = pd.DataFrame(x_test, columns=x_names)

# Group features for aggregation
# n_labels_fb, feature_groups_fb = group_freq_bands_shap(x_names)

# n_labels, feature_groups = group_methods_shap(x_names)
#
# # Calculate aggregated SHAP values for each feature group
# grouped_shap_values = np.zeros((len(x_test), len(n_labels)))
# for i, group in enumerate(feature_groups):
#     grouped_shap_values[:, i] = np.sum(shap_values[:, group], axis=1)
#
# shap.initjs()
# shap.summary_plot(grouped_shap_values, feature_names=n_labels)
# # shap.summary_plot(shap_values, x_test_df, max_display=100)
# plt.show()

montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
ch_pos = montage.get_positions()['ch_pos']
ch_pos.pop('Cz')
# ch_names = list(ch_pos.keys())
ch_names = chan_map.keys()

freq_bands = {
    'delta': [0.5, 4],
    'theta': [4, 7],
    'alpha': [7, 13],
    'beta': [13, 30],
    'whole_spec': [0.5, 30]
}
for freq_band in freq_bands.keys():
    n_labels, feature_groups = group_chan_fb(x_names, ch_names, freq_band)

    # Calculate aggregated SHAP values for each feature group
    grouped_shap_values = np.zeros((len(x_test), len(n_labels)))
    for i, group in enumerate(feature_groups):
        grouped_shap_values[:, i] = np.sum(shap_values[:, group], axis=1)
    vals = np.abs(grouped_shap_values).mean(0)
    # shap.initjs()
    # shap.summary_plot(grouped_shap_values, feature_names=n_labels)
    imp = []
    upper_q = np.quantile(vals, 0.90)
    print(min(vals), max(vals), upper_q)
    for i in vals:
        if i > upper_q:
            imp.append(upper_q)
        else:
            imp.append(i)
    band_r = freq_bands[freq_band]
    title = f"{freq_band} ({band_r[0]} - {band_r[1]} Hz)"
    if freq_band == 'whole_spec':
        title = f"whole spectrum ({band_r[0]} - {band_r[1]} Hz)"
    plot_topo_vals_12(imp, title)
    plt.tight_layout()
    plt.savefig(f'topo_shap_{freq_band}')
    plt.show()
