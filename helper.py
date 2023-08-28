import json
import pickle

import numpy as np

from config import freq_bands, methods


def load_object(fname):
    try:
        with open(fname + ".pickle", "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


def save_object(obj, fname):
    try:
        with open(fname + ".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


def str_to_dict(param):
    param = param.replace('OrderedDict', '')
    param = param.replace('([', '{')
    param = param.replace('])', '}')
    param = param.replace(',', ':')
    param = param.replace(')', ', ')
    param = param.replace('(', '')
    param = param.replace("'", '"')
    param = param.replace(", }", '}')
    param = json.loads(param)
    return param


def group_freq_bands_shap(x_names):
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
    return n_labels_fb, feature_groups_fb


def group_methods_shap(x_names):
    fg = []
    n_labels = []
    for m in methods:
        feature_group_idx = []
        for i in range(len(x_names)):
            if m in x_names[i]:
                feature_group_idx.append(i)
        if len(feature_group_idx) > 0:
            fg.append(feature_group_idx)
            n_labels.append(m)
    return n_labels, fg


def equalize_classes(targets, threshold=3):
    l = np.unique(np.array(targets), return_counts=True)
    tar_map = []
    for i in range(len(l[0])):
        if l[1][i] < threshold:
            c_up = i
            c_down = i
            con = True
            con2 = False
            while con:
                if c_up + 1 < len(l[0]):
                    c_up += 1
                    if l[1][c_up] >= threshold:
                        tar_map.append([l[0][i], l[0][c_up]])
                        con = False

                if c_down - 1 >= 0 and con:
                    c_down -= 1
                    if l[1][c_down] >= threshold:
                        tar_map.append([l[0][i], l[0][c_down]])
                        con = False

    tar_map_t = np.array(tar_map).transpose()
    print(tar_map)
    y_new = []
    for age in targets:
        if age in tar_map_t[0]:
            id_age = list(tar_map_t[0]).index(age)
            y_new.append(tar_map_t[1][id_age])
        else:
            y_new.append(age)
    return y_new
