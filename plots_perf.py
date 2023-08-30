import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

from helper import load_object


MODEL_LIST = [
    'SVRegression', 'MLP', 'KernalRige', 'KNN', 'BaggedKNN', 'LassoRegression', 'EleasticNet', 'RandomForrest',

]
for m in MODEL_LIST:
    obj = load_object(m+'/best_model')
    if obj is None:
        continue
    print(m)
    score = []
    rs = []
    for key in obj.keys():
        if 'fold' in key:
            print(obj[key])
            score.append(obj[key][0])
            rs.append(obj[key][1])
    print(np.mean(score), np.mean(rs))
sys.exit()


SET_PATH = r'/home/tobias/Schreibtisch/EEG-FeatureExtraction/trainingSets/TSFinal/'

set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']
train_set = ['TS2/']
results = dict(
    mean=[],
    std=[],
    label=[]
)

for m in MODEL_LIST:
    sets_mean, sets_std = [], []
    for ts in train_set:
        for sv in set_vary:
            f_name = ts.replace('/', '_') + sv.replace('/', '_')
            res_df = pd.read_csv(f'{m}/{f_name}results.csv')
            res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
            best_params = res_df.iloc[0]
            best_mean = best_params['mean_test_score']
            best_std = best_params['std_test_score']
            if best_mean < 0:
                best_mean = -1 * best_mean
            if float(best_mean) > 5:
                best_mean = float(best_mean) / 10
                best_std = float(best_std) / 10

            sets_mean.append(best_mean)
            sets_std.append(best_std)
    results['mean'].append(sets_mean)
    results['std'].append(sets_std)
    results['label'].append(m)

sets_name = []
for ts in train_set:
    for sv in set_vary:
        set_name = 'set-1' if 'TS4' in ts else 'set-2'
        if 'EC' in sv:
            set_name += '-EC'
        elif 'EO' in sv:
            set_name += '-EO'
        sets_name.append(set_name)

mean_perf = np.array(results['mean']).transpose()
std_perf = np.array(results['std']).transpose()
dif_mean = [i[0] - i[1] for i in results['mean']]
mean_d = np.mean(dif_mean)
print(dif_mean)
print(mean_d)

fig, ax = plt.subplots(figsize=(5, 6))

sets_per_model = len(sets_name)
space_per_model = sets_per_model * 0.5
sep_space = 0.5
label_offset = ((sets_per_model - 1) / 2) * 0.5
x_pos = np.linspace(1, len(MODEL_LIST) * (space_per_model + sep_space), len(MODEL_LIST))
label_pos = [i + label_offset for i in x_pos]

for i in range(len(mean_perf)):
    x_scale = i * 0.5
    ax.barh(
        x_pos + x_scale, mean_perf[i],
        xerr=std_perf[i],
        align='center', alpha=0.7, ecolor='red',
        capsize=3, height=0.4, label=sets_name[i])

# Create custom error cap markers
error_cap_marker = mlines.Line2D([], [], color='red', marker='_', markersize=8, markeredgewidth=2, label='std fold')

# Combine legend handles and labels
handles, labels = ax.get_legend_handles_labels()
handles.append(error_cap_marker)
labels.append(error_cap_marker.get_label())

ax.set_yticks(label_pos)
ax.set_yticklabels(MODEL_LIST)
ax.set_xlim(1.4, 3.5)
ax.legend(handles, labels, loc='upper right')
ax.set_xlabel('Mean Absolute Error')
plt.tight_layout()

fname = ''
for n in sets_name:
    fname += n + '_'
fname += 'performance'
plt.savefig(fname)
plt.show()
