from helper import load_object

BASE_PATH = r'/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge/'
# r'/home/tobias/Schreibtisch/ML_Models_BrainAge/'
SET_PATH = r'/scratch/modelrep/sadiya/students/tobias/train_sets/'
# r'/home/tobias/Schreibtisch/EEG-FeatureExtraction/trainingSets/TSFinal/'


chan_map = {
    "R1": ["E17", "E14", "E21", "E22", "E18", "E15", "E16", "E11", "E10", "E9"],
    "R2": ["E20", "E12", "E5", "E118", "E13", "E6", "E112", "E7", "E106"],
    "R3": ["E55", "E54", "E79", "E53", "E86", "E61", "E78", "E62", "E67", "E77", "E72"],
    "R4": ["E71", "E76", "E70", "E75", "E83", "E69", "E74", "E82", "E89", "E73", "E81", "E88"],
    "R5": ["E127", "E25", "E23", "E19", "E128", "E32", "E26", "E27", "E24"],
    "R6": ["E38", "E33", "E48", "E43", "E34", "E44", "E39", "E49", "E40", "E45", "E56"],
    "R7": ["E28", "E29", "E35", "E30", "E36", "E41", "E31", "E42", "E37", "E46", "E47"],
    "R8": ["E50", "E51", "E52", "E57", "E60", "E58", "E59", "E63", "E64", "E65", "E66", "E68"],
    "R9": ["E126", "E8", "E3", "E4", "E2", "E1", "E124", "E123", "E125"],
    "R10": ["E122", "E121", "E116", "E120", "E119", "E109", "E114", "E115", "E113", "E108", "E107"],
    "R11": ["E117", "E110", "E111", "E105", "E104", "E103", "E80", "E87", "E93", "E98", 'E102'],
    "R12": ["E92", "E97", "E101", "E100", "E85", "E91", "E96", "E84", "E90", "E95", "E99", "E94"],
}
methods = [
    'pow_freq_bands',
    'svd_fisher_info',
    'hjorth_complexity_spect',
    'wavelet_coef_energy',
    'hjorth_complexity',
    'spect_slope',
    'std',
    'ptp_amp',
    'quantile',
    'line_length',
    'zero_crossings',
    'skewness',
    'kurtosis',
    'higuchi_fd',
    'samp_entropy',
    'app_entropy',
    'spect_entropy',
    'mean',
    'hurst_exp'
]
freq_bands = ['delta', 'theta', 'alpha', 'beta', 'whole_spec']


methods = [
    'pow_freq_bands',
    'svd_fisher_info',
    'hjorth_complexity_spect',
    'wavelet_coef_energy',
    'hjorth_complexity',
    'spect_slope',
    'std',
    'ptp_amp',
    'quantile',
    'line_length',
    'zero_crossings',
    'skewness',
    'kurtosis',
    'higuchi_fd',
    'samp_entropy',
    'app_entropy',
    'spect_entropy',
    'mean',
    'hurst_exp'
]

training_sets = ['TS2/']
set_vary = []

set_path = SET_PATH + 'TS2/' + 'meanEpochs/'
data = load_object(set_path + 'training_set')
x_names = data['x_names']
for m in methods:
    exsis = False
    for xlab in x_names:
        if m in xlab:
            exsis = True
            break
    if not(exsis):
        print(m)