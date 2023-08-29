import pandas as pd
from matplotlib import pyplot as plt

MODEL_LIST = [
    'EleasticNet',
    'KNN', 'LassoRegression',
    'MLP', 'SVRegression'
]

SET_PATH = r'/home/tobias/Schreibtisch/EEG-FeatureExtraction/trainingSets/TSFinal/'

result_df = dict(
    model=[],
    fold_scores=[],
    average=[]
)

ts = 'TS2/'
sv = 'meanEpochs/'

set_vary = ['meanEpochs/', 'meanEpochs/onlyEC/', 'meanEpochs/onlyEO/']

results = dict(
    mean=[],
    std=[],
    label=[]
)

for m in MODEL_LIST:
    f_name = ts.replace('/', '_') + sv.replace('/', '_')
    res_df = pd.read_csv(f'{m}/{f_name}results.csv')
    res_df = res_df.sort_values(by=['mean_test_score'], ascending=False)
    best_params = res_df.iloc[0]
    print(best_params)
    best_mean = best_params['mean_test_score']
    best_std = best_params['std_test_score']
    if best_mean < 0:
        best_mean = -1*best_mean
    if float(best_mean) > 5:
        best_mean = float(best_mean)/10
        best_std = float(best_std)/10

    results['mean'].append(best_mean)
    results['std'].append(best_std)
    results['label'].append(m)

fig, ax = plt.subplots()
x_pos = [i for i in range(1, len(MODEL_LIST)+1)]

ax.bar(x_pos, results['mean'], yerr=results['std'], align='center', alpha=0.7, ecolor='red', capsize=3, width=0.4)
ax.set_xticks(x_pos)
ax.set_xticklabels(MODEL_LIST)
plt.show()

