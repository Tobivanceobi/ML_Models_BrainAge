import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from config import freq_bands
from helper import load_object, group_freq_bands_shap, group_methods_shap

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

for m in MODEL_LIST:
    print(m)
    set_path = SET_PATH + 'TS2/meanEpochs/'
    data = load_object(set_path + 'training_set')
    x = data['x']
    groups = data['group']
    y = data['y']
    x_names = data['x_names']
    y = [int(age * 10) for age in data['y']]


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
    # n_labels_fb, feature_groups_fb = group_freq_bands_shap(x_names)

    n_labels_m, feature_groups_m = group_methods_shap(x_names)

    # Calculate aggregated SHAP values for each feature group
    grouped_shap_values = np.zeros((len(x_test), len(n_labels_m)))
    for i, group in enumerate(feature_groups_m):
        grouped_shap_values[:, i] = np.sum(shap_values[:, group], axis=1)

    vals = np.abs(grouped_shap_values).mean(0)
    vals, n_labels = zip(*sorted(zip(vals, n_labels_m), reverse=True))
    print(n_labels)

    result_df['feature_rank'].append(list(n_labels))
    result_df['model'].append(m)
    result_df['shap_vals'].append(vals)

le = LabelEncoder()
le.fit(result_df['feature_rank'][0])
ranks = []
for k in result_df['feature_rank']:
    ranks.append(le.transform(k))
print(ranks)

correlation_matrix = []
for x in ranks:
    r = []
    for l in ranks:
        coef, p = spearmanr(x, l)
        # coef, p = kendalltau(x, l)
        r.append(coef)
    correlation_matrix.append(r)
corr_mat = np.array(correlation_matrix)

# Calculate the distance matrix from the correlation matrix
distance_matrix = np.sqrt(2 * (1 - corr_mat))

# Perform hierarchical clustering
linkage_matrix = hierarchy.linkage(distance_matrix, method='complete')

# Reorder the rows and columns based on clustering
dendrogram = hierarchy.dendrogram(linkage_matrix, no_plot=True)
reordered_indices = dendrogram['leaves']

sorted_corr_mat = corr_mat[reordered_indices][:, reordered_indices]

print(sorted_corr_mat)
plt.figure(figsize=(8, 6))
sns.heatmap(sorted_corr_mat, annot=True, cmap='coolwarm', xticklabels=result_df['model'],
            yticklabels=result_df['model'])
plt.title('Correlation between Feature Group SHAP Values')
plt.savefig('method_groups.png')
plt.show()
