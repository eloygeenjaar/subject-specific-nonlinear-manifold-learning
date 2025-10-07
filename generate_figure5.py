import numpy as np
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from nilearn import masking
from nilearn import plotting
from sklearn.decomposition import FastICA
from scipy.stats import ttest_ind, false_discovery_control


# From: https://stackoverflow.com/questions/35668219/how-to-set-up-a-custom-font-with-custom-path-to-matplotlib-global-font
fe = fm.FontEntry(
    fname='/data/users1/egeenjaar/fonts/montserrat/static/Montserrat-SemiBold.ttf',
    name='Montserrat')
fm.fontManager.ttflist.append(fe)

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = fe.name

mask_img = nb.load('images/fbirn_mask.nii')

data = np.load('results/group_maps.npy')
sz_mask = np.load('results/group_maps_sz.npy')
print(data.shape)
num_subjects, num_dimensions, num_voxels = data.shape
sz_mask_rep = np.repeat(sz_mask[:, np.newaxis], num_dimensions, axis=1).flatten()
subjects = np.repeat(np.arange(num_subjects)[:, np.newaxis], num_dimensions, axis=1).flatten()
dimensions = np.repeat(np.arange(num_dimensions)[np.newaxis], num_subjects, axis=0).flatten()
ica = FastICA(n_components=64, max_iter=1000, random_state=42)
cut_coords = [-30, -5, 20, 45]
data_t = np.reshape(data, (num_subjects * num_dimensions, num_voxels)).T

comp_p = Path('results/ica_components.npy')
mean_p = Path('results/ica_mean.npy')

if comp_p.is_file() and mean_p.is_file():
    components = np.load(comp_p)
    mean = np.load(mean_p)
    sources = np.dot(data_t - mean, components.T)
else:
    sources = ica.fit_transform(data_t)
    np.save(comp_p, ica.components_)
    np.save(mean_p, ica.mean_)
    components = ica.components_
    mean = ica.mean_

data_t = data_t - mean

result = ttest_ind(components.T[sz_mask_rep], components.T[~sz_mask_rep])
pvals = false_discovery_control(result.pvalue)
print(result.statistic)

statistic = result.statistic * (pvals <= 0.05)

#for i in range(64):
#    if np.abs(statistic[i]) > 0:
#        print(f'ICA component: {i}, p-value: {np.format_float_scientific(pvals[i], precision=2)}, statistic: {np.abs(statistic[i])}')
#        fig, ax = plt.subplots(1, 1)
#        sources[:, i] = (np.abs(sources[:, i]) >= 0.2 * np.abs(sources[:, i]).max()) * sources[:, i]
#        
#        plotting.plot_stat_map(
#            img,
#            axes=ax, cmap='jet', display_mode='z', alpha=0.6, cut_coords=cut_coords, figure=fig,
#            annotate=False, colorbar=True)
#        ax.set_title(f'Statistic: {round(float(statistic[i]), 2)}')
#        #plotting.plot_stat_map(
#        #    masking.unmask((diff), mask_img),
#        #    axes=axs[1], cmap='jet', display_mode='z', alpha=0.6, cut_coords=cut_coords, figure=fig,
#        #    annotate=False, colorbar=True)
#        #axs[1].set_title('SZ - C t-test FDR corrected')
#        plt.savefig(f'figures_ica/ica_{i}.png', dpi=400)
#        plt.clf()
#        plt.close(fig)#

#        nb.save(img, f'figures_ica/ica_{i}.nii')

ica_components = [4, 15, 21, 25, 27, 41, 59, 61]
for (i, ica_component_ix) in enumerate(ica_components):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    print(f'ICA component: {ica_component_ix}, p-value: {np.format_float_scientific(pvals[ica_component_ix], precision=2)}, statistic: {np.abs(statistic[ica_component_ix])}')
    sources[:, ica_component_ix] = (np.abs(sources[:, ica_component_ix]) >= 0.2 * np.abs(sources[:, ica_component_ix]).max()) * sources[:, ica_component_ix]
    img = masking.unmask((sources[:, ica_component_ix]) * np.sign(statistic)[ica_component_ix], mask_img)
    plotting.plot_stat_map(
        img,
        axes=ax, cmap='jet', display_mode='z', alpha=0.6, cut_coords=cut_coords, figure=fig,
        annotate=False, colorbar=False)
    plt.savefig(f'figures_ica_presentation/{i}.png', dpi=400)
    plt.clf()
    plt.close(fig)