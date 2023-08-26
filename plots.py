from helper import load_object

MODEL_LIST = [
    'BaggedKNN', 'CatBoost', 'EleasticNet',
    'KNN', 'LassoRegression', 'LogisticRegression',
    'MLP', 'RandomForrest', 'SVRegression', 'XGBoost'
]

for m in MODEL_LIST:

    shap_dict = load_object(m + '/' + 'shap_values')
    print(shap_dict.keys())
    break
