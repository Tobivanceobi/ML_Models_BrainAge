BASE_PATH = r'/home/modelrep/sadiya/tobias_ettling/ML_Models_BrainAge/'
# r'/home/tobias/Schreibtisch/ML_Models_BrainAge/'
SET_PATH = r'/scratch/modelrep/sadiya/students/tobias/train_sets/'
# r'/home/tobias/Schreibtisch/EEG-FeatureExtraction/trainingSets/TSFinal/'

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