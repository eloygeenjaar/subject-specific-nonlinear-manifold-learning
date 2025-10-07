import numpy as np
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from pathlib import Path
from nilearn import masking
from nilearn import plotting
from sklearn.decomposition import FastICA
from scipy.stats import ttest_ind, false_discovery_control
from scipy.stats import pearsonr, f_oneway



dose = np.load('results/dose.npy', allow_pickle=True)
print(dose)
cminds = np.load('results/cminds.npy', allow_pickle=True)
cminds_mask = np.load('results/cminds_mask.npy', allow_pickle=True)
#cminds_mask = cminds_mask & 
nan_mask = ((cminds != -9999) & (~np.isnan(cminds)))
nan_mask_dose = ((dose != -9999) & (~np.isnan(dose)))

mask_img = nb.load('images/fbirn_mask.nii')

data = np.load('results/group_maps.npy')
sz_mask = np.load('results/group_maps_sz.npy')
print(data.shape)
num_subjects, num_dimensions, num_voxels = data.shape
sz_mask_rep = np.repeat(sz_mask[:, np.newaxis], num_dimensions, axis=1).flatten()
cminds_mask_rep = np.repeat(cminds_mask[:, np.newaxis], num_dimensions, axis=1).flatten()
cminds_rep = np.repeat(cminds[:, np.newaxis], num_dimensions, axis=1).flatten()
nan_mask_rep = np.repeat(nan_mask[:, np.newaxis], num_dimensions, axis=1).flatten()
dose_rep = np.repeat(dose[:, np.newaxis], num_dimensions, axis=1).flatten()
nan_mask_dose_rep = np.repeat(nan_mask_dose[:, np.newaxis], num_dimensions, axis=1).flatten()
subjects = np.repeat(np.arange(num_subjects)[:, np.newaxis], num_dimensions, axis=1).flatten()
dimensions = np.repeat(np.arange(num_dimensions)[np.newaxis], num_subjects, axis=0).flatten()
ica = FastICA(n_components=64, max_iter=1000, random_state=42)
cut_coords = [-30, -5, 20, 45]
data_t = np.reshape(data, (num_subjects * num_dimensions, num_voxels)).T
print(data_t.shape)

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

print('CMINDS')
cminds_stats, cminds_pvals = np.zeros((64, )), np.zeros((64, ))
for i in range(64):
    r, p = pearsonr(components.T[cminds_mask_rep][nan_mask_rep][:, i], cminds_rep[nan_mask_rep])
    cminds_stats[i] = r
    cminds_pvals[i] = p
    print(r)
print(components.T[cminds_mask_rep][nan_mask_rep].shape, cminds_rep[nan_mask_rep].shape)

cminds_pvals = false_discovery_control(cminds_pvals)
cminds_stats = cminds_stats * (cminds_pvals <= 0.05)

print('CPZ DOSE')
dose_stats, dose_pvals = np.zeros((64, )), np.zeros((64, ))
for i in range(64):
    r, p = pearsonr(components.T[cminds_mask_rep][nan_mask_dose_rep][:, i], dose_rep[nan_mask_dose_rep])
    dose_stats[i] = r
    dose_pvals[i] = p
    print('dose', p, r)

dose_pvals = false_discovery_control(dose_pvals)
dose_stats = dose_stats * (dose_pvals <= 0.05)

print(dose_stats)

print(dose_stats[[4, 15, 21, 25, 27, 41, 59, 61]])

print(components.T.shape)
print(components.shape)
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

cmap = mc.LinearSegmentedColormap.from_list("newcmap",
    [
        mc.hex2color("#83BAF2"),
        mc.hex2color("#FFFFFF"),
        mc.hex2color("#8776B3")
    ]
)

ica_component_ixs = [4, 15, 21, 25, 27, 41, 59, 61]

for ica_i in ica_component_ixs:
    print(f'ICA component: {ica_i} CMINDS p-value: {np.format_float_scientific(cminds_pvals[ica_i], precision=2)}, statistic: {cminds_stats[ica_i] * np.sign(statistic)[ica_i]}')
    print(f'ICA component: {ica_i} Dose p-value: {np.format_float_scientific(dose_pvals[ica_i], precision=2)}, statistic: {dose_stats[ica_i] * np.sign(statistic)[ica_i]}')

for ica_component_ix in ica_component_ixs:
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    print(f'ICA component: {ica_component_ix}, p-value: {np.format_float_scientific(pvals[ica_component_ix], precision=2)}, statistic: {np.abs(statistic[ica_component_ix])}')
    vminmax = np.max(np.abs(sources[:, ica_component_ix]))
    sources[:, ica_component_ix] = (np.abs(sources[:, ica_component_ix]) >= (0.1 * vminmax)) * sources[:, ica_component_ix]

    img = masking.unmask((sources[:, ica_component_ix]) * np.sign(statistic)[ica_component_ix], mask_img)
    plotting.plot_stat_map(
        img,
        axes=ax, cmap='bwr', display_mode='z', alpha=0.8, cut_coords=cut_coords, figure=fig,
        annotate=False, colorbar=False)
    plt.savefig(f'figures_ica_presentation/ica_{ica_component_ix}.png', dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close(fig)
