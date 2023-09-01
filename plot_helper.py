import mne
import numpy as np
from matplotlib import pyplot as plt

from config import methods, freq_bands, chan_map


def plot_topo_vals_12(vals, title):
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    ch_pos = montage.get_positions()['ch_pos']
    ch_pos.pop('Cz')
    points_3d = np.array([ch_pos[key] for key in ch_pos.keys()])
    ch_names = list(ch_pos.keys())

    # Normalize points to create a sphere
    points_norm = points_3d / np.linalg.norm(points_3d, axis=1, keepdims=True)

    # Project points using azimuthal equidistant projection
    radius = 1.5  # Sphere radius
    projection_scale = 0.07  # Scale factor for projection
    x_2d = projection_scale * radius * points_norm[:, 0] / (points_norm[:, 2] + radius)
    y_2d = projection_scale * radius * points_norm[:, 1] / (points_norm[:, 2] + radius)

    cpos = np.array([x_2d, y_2d]).transpose()

    plot_cord = {key: [] for key in chan_map.keys()}

    for i in range(0, len(ch_names)):
        for reg in chan_map.keys():
            if ch_names[i] in chan_map[reg]:
                plot_cord[reg].append(cpos[i])

    region_cord = {key: [] for key in chan_map.keys()}
    for key in plot_cord.keys():
        points = np.array(plot_cord[key]).transpose()
        region_cord[key] = [sum(points[0]) / len(points[0]), sum(points[1]) / len(points[1])]

    reg_pos = np.array([region_cord[key] for key in region_cord.keys()])
    fig, ax = plt.subplots(figsize=(8, 8))
    mne.viz.plot_topomap(vals, pos=reg_pos, show=False, axes=ax, cmap='viridis')

    cbar = plt.colorbar(ax.get_images()[0], ax=ax, orientation='horizontal', pad=0.05)
    cbar.ax.tick_params(labelsize=50)
    cbar.set_label('Shap Importance', rotation=270, labelpad=60, fontsize=20)
    plt.title(title, fontsize=50)


def plot_topo_vals_128(eeg_data, title):
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

    ch_pos = montage.get_positions()['ch_pos']
    ch_pos.pop('Cz')
    points_3d = np.array([ch_pos[key] for key in ch_pos.keys()])

    # Normalize points to create a sphere
    points_norm = points_3d / np.linalg.norm(points_3d, axis=1, keepdims=True)

    # Project points using azimuthal equidistant projection
    radius = 1.5  # Sphere radius
    projection_scale = 0.07  # Scale factor for projection
    x_2d = projection_scale * radius * points_norm[:, 0] / (points_norm[:, 2] + radius)
    y_2d = projection_scale * radius * points_norm[:, 1] / (points_norm[:, 2] + radius)

    cpos = np.array([x_2d, y_2d]).transpose()

    fig, ax = plt.subplots(figsize=(8, 8))
    mne.viz.plot_topomap(eeg_data, pos=cpos, show=False, axes=ax, cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(ax.get_images()[0], ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('EEG Data (Mean)', rotation=270, labelpad=15, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    plt.title(title, fontsize=50)


def group_freq_bands_shap(x_names):
    feature_groups_fb = []
    n_labels_fb = []
    for fb in freq_bands:
        feature_group_idx = []
        for i in range(len(x_names)):
            if fb in x_names[i]:
                if fb == 'whole_spec':
                    temp = True
                    for ofb in ['delta', 'theta', 'alpha', 'beta']:
                        if ofb in x_names[i]:
                            temp = False
                    if temp:
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


def group_pow_shap(x_names, chans, fb):
    fg = []
    n_labels = []
    for chan in chans:
        feature_group_idx = []
        for i in range(len(x_names)):
            if fb in x_names[i] and chan in x_names[i] and 'pow' in x_names[i]:
                feature_group_idx.append(i)
                print('ok')
        if len(feature_group_idx) > 0:
            fg.append(feature_group_idx)
            n_labels.append(chan + '_pow_freq_' + fb)
    return n_labels, fg


def group_chan_fb(x_names, chans, fb):
    fg = []
    n_labels = []
    for chan in chans:
        feature_group_idx = []
        for i in range(len(x_names)):
            if fb in x_names[i] and chan in x_names[i]:
                if fb == 'whole_spec':
                    temp = True
                    for ofb in ['delta', 'theta', 'alpha', 'beta']:
                        if ofb in x_names[i]:
                            temp = False
                    if temp:
                        feature_group_idx.append(i)
                else:
                    feature_group_idx.append(i)
        if len(feature_group_idx) > 0:
            fg.append(feature_group_idx)
            n_labels.append(chan)
    return n_labels, fg
