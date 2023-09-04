import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

from helper import load_object


MODEL_LIST = [
    'KNN', 'BaggedKNN', 'LassoRegression', 'EleasticNet', 'SVRegression', 'RandomForrest',
    'XGBoost'
]



SET_PATH = r'/home/tobias/Schreibtisch/EEG-FeatureExtraction/trainingSets/TSFinal/'

set_vary = [''] # ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
train_set = ['TS5/'] # ['TS2/', 'TS4/']
results = dict(
    mean=[],
    std=[],
    label=[]
)

for m in MODEL_LIST:
    sets_mean, sets_std = [], []
    f_name = train_set[0].replace('/', '_')
    res_df = pd.read_csv(f'{m}/{f_name}results.csv')
    res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
    best_params = res_df.iloc[0]
    best_mean = best_params['mean_test_score']
    best_std = best_params['std_test_score']
    if best_mean < 0:
        best_mean = -1 * best_mean
    if float(best_mean) > 15:
        best_mean = float(best_mean) / 10
        best_std = float(best_std) / 10

    sets_mean.append(best_mean)
    sets_std.append(best_std)
    results['mean'].append(sets_mean)
    results['std'].append(sets_std)
    results['label'].append(m)


mean_perf = np.array(results['mean']).transpose()
std_perf = np.array(results['std']).transpose()

# dif_mean = [abs(i[0] - i[1]) for i in results['mean']]
# mean_d = np.mean(dif_mean)
# print(dif_mean)
# print(mean_d)

model_labels = []
for lab_m in MODEL_LIST:
    if lab_m == 'LassoRegression':
        model_labels.append('Lasso')
    else:
        model_labels.append(lab_m)

model_labels.append('FIN-Ensemble')
mean_perf = np.append(mean_perf[0], 2.33)

print(model_labels)
print(mean_perf)

fig, ax = plt.subplots(figsize=(5, 6))

sets_per_model = 1
space_per_model = sets_per_model * 0.5
sep_space = 0.5
label_offset = ((sets_per_model - 1) / 2) * 0.5
x_pos = np.linspace(1, len(model_labels) * (space_per_model + sep_space), len(model_labels))
label_pos = [i + label_offset for i in x_pos]


ax.barh(
    x_pos, mean_perf,
    align='center', alpha=0.7, height=0.4)


ax.set_yticks(label_pos)
ax.set_yticklabels(model_labels)
ax.set_xlim(1.4, 3.5)
ax.set_xlabel('Mean Absolute Error')
plt.tight_layout()


fname = 'TS5_performance'
plt.savefig(fname)
plt.show()
