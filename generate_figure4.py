import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def kde_func(df, sz_mask, num=200):
    sz_values = df.loc[sz_mask, ["x", "y"]].values.T
    c_values = df.loc[~sz_mask, ["x", "y"]].values.T

    xmin = min(np.min(sz_values[0]), np.min(c_values[0])) - 1.
    xmax = max(np.max(sz_values[0]), np.max(c_values[0])) + 1.
    ymin = min(np.min(sz_values[1]), np.min(c_values[1])) - 1.
    ymax = max(np.max(sz_values[1]), np.max(c_values[1])) + 1.

    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, num),
        np.linspace(ymin, ymax, num), indexing='xy')
    positions = np.vstack([X.ravel(), Y.ravel()])

    sz_kernel = gaussian_kde(sz_values)
    c_kernel = gaussian_kde(c_values)

    Z_sz = np.reshape(sz_kernel(positions).T, X.shape)
    Z_c = np.reshape(c_kernel(positions).T, X.shape)

    Z = Z_sz - Z_c

    return Z, (xmin, xmax, ymin, ymax)


# From: https://stackoverflow.com/questions/35668219/how-to-set-up-a-custom-font-with-custom-path-to-matplotlib-global-font
fe = fm.FontEntry(
    fname='/data/users1/egeenjaar/fonts/montserrat/static/Montserrat-SemiBold.ttf',
    name='Montserrat')
fm.fontManager.ttflist.append(fe)

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = fe.name

subject_weight_results = np.load('results/subject_weights_results.npz')

train_pca_weights = subject_weight_results['train_pca_weights']
test_pca_weights = subject_weight_results['test_pca_weights']

pca_weights = np.concatenate((
    train_pca_weights,
    test_pca_weights),
    axis=0
)

train_mask = subject_weight_results['train_mask']
test_mask = subject_weight_results['test_mask']

mask = np.concatenate((
    train_mask,
    test_mask),
    axis=0    
)

train_sz_mask = subject_weight_results['train_sz_mask']
test_sz_mask = subject_weight_results['test_sz_mask']

sz_mask = np.concatenate((
    train_sz_mask,
    test_sz_mask),
    axis=0
)

train_df = pd.DataFrame(
    np.zeros((train_mask.sum(), 3)),
    columns=['x', 'y', 'diagnosis']
)
train_df['diagnosis'] = train_df['diagnosis'].astype(str)
train_df.loc[:, 'x'] = train_pca_weights[train_mask][:, 0]
train_df.loc[:, 'y'] = train_pca_weights[train_mask][:, 1]
train_df.loc[train_sz_mask, 'diagnosis'] = 'sz'
train_df.loc[~train_sz_mask, 'diagnosis'] = 'c'

test_df = pd.DataFrame(
    np.zeros((test_mask.sum(), 3)),
    columns=['x', 'y', 'diagnosis']
)
test_df['diagnosis'] = test_df['diagnosis'].astype(str)
test_df.loc[:, 'x'] = test_pca_weights[test_mask][:, 0]
test_df.loc[:, 'y'] = test_pca_weights[test_mask][:, 1]
test_df.loc[test_sz_mask, 'diagnosis'] = 'sz'
test_df.loc[~test_sz_mask, 'diagnosis'] = 'c'

df = pd.DataFrame(
    np.zeros((mask.sum(), 3)),
    columns=['x', 'y', 'diagnosis'])
df['diagnosis'] = df['diagnosis'].astype(str)
df.loc[:, 'x'] = pca_weights[mask][:, 0]
df.loc[:, 'y'] = pca_weights[mask][:, 1]
df.loc[sz_mask, 'diagnosis'] = 'sz'
df.loc[~sz_mask, 'diagnosis'] = 'c'
print(df)

Z_train, (train_xmin, train_xmax, train_ymin, train_ymax) = kde_func(train_df, train_sz_mask, num=200)
Z_test, (test_xmin, test_xmax, test_ymin, test_ymax) = kde_func(test_df, test_sz_mask, num=200)

xmin = min(train_xmin, test_xmin)
xmax = max(train_xmax, test_xmax)
ymin = min(train_ymin, test_ymin)
ymax = max(train_ymax, test_ymax)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].imshow(Z_train, cmap='bwr',
          extent=[xmin, xmax, ymin, ymax], alpha=0.8, aspect=(1.0/2))
axs[1].imshow(Z_test, cmap='bwr',
          extent=[xmin, xmax, ymin, ymax], alpha=0.8, aspect=(1.0/2))
axs[0].set_title('Training SZ - C distribution')
axs[1].set_title('Test SZ - C distribution')
axs[1].set_xlabel('PCA component 1')
axs[0].set_xlabel(None)
axs[0].set_ylabel('PCA component 2')
axs[1].set_ylabel('PCA component 2')
axs[0].set_xticks([])
axs[0].set_xticklabels([])
plt.tight_layout()
plt.savefig('figures/figure4b.png', dpi=400)

seeds = [42, 1337, 9999, 1212]

group_results = {
    42: {0.01: 26.934972763061523, 0.05: 26.862558364868164, 0.1: 26.925331115722656, 0.2: 27.20204734802246, 0.3: 27.403568267822266, 0.4: 27.36562728881836, 0.5: 27.533552169799805},
    1337: {0.01: 27.036754608154297, 0.05: 26.966064453125, 0.1: 27.02430534362793, 0.2: 27.312397003173828, 0.3: 27.526762008666992, 0.4: 27.489337921142578, 0.5: 27.667924880981445 },
    9999: {0.01: 26.846294403076172, 0.05: 26.77190589904785, 0.1: 26.834156036376953, 0.2: 27.113927841186523, 0.3: 27.33153533935547, 0.4: 27.297319412231445, 0.5: 27.47330665588379},
    1212: {0.01: 27.070215225219727, 0.05: 26.99213981628418, 0.1: 27.051414489746094, 0.2: 27.336606979370117, 0.3: 27.54482650756836, 0.4: 27.508277893066406, 0.5: 27.68716812133789}
}
decomposed_results = {
    42: {0.01: 26.41666030883789, 0.05: 26.03294563293457, 0.1: 25.334095001220703, 0.2: 24.722301483154297, 0.3: 24.555126190185547, 0.4: 24.353918075561523, 0.5: 24.34992790222168},
    1337: {0.01: 26.467880249023438, 0.05: 26.20921516418457, 0.1: 25.463171005249023, 0.2: 24.85824966430664, 0.3: 24.681529998779297, 0.4: 24.459205627441406, 0.5: 24.441598892211914},
    9999: {0.01: 26.345775604248047, 0.05: 25.886932373046875, 0.1: 25.187681198120117, 0.2: 24.60325813293457, 0.3: 24.46377944946289, 0.4: 24.27650260925293, 0.5: 24.282297134399414},
    1212: {0.01: 26.419353485107422, 0.05: 26.062620162963867, 0.1: 25.346885681152344, 0.2: 24.77719497680664, 0.3: 24.634164810180664, 0.4: 24.41835594177246, 0.5: 24.433673858642578}
}

palette = {
        'Group': '#B8E3FF',
        'Decomposed': '#8776B3',
        #  #83BAF2- blue
        #3981B8 - dark blue
    }

df = pd.DataFrame(
    np.zeros((4 * 7 * 2, 4)),
    columns=['seed', 'perc', 'model', 'val']
)
ix = 0
for (seed_key, seed_dict) in group_results.items():
    for (perc_key, perc_val) in seed_dict.items():
        df.loc[ix, 'seed'] = seed_key
        df.loc[ix, 'perc'] = perc_key
        df.loc[ix, 'model'] = 'Group'
        df.loc[ix, 'val'] = perc_val
        ix += 1

for (seed_key, seed_dict) in decomposed_results.items():
    for (perc_key, perc_val) in seed_dict.items():
        df.loc[ix, 'seed'] = seed_key
        df.loc[ix, 'perc'] = perc_key
        df.loc[ix, 'model'] = 'Decomposed'
        df.loc[ix, 'val'] = perc_val
        ix += 1

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(df, x='perc', y='val', hue='model', errorbar='sd', palette=palette, legend=False, linewidth=5)
ax.set_xlabel('Percentage of timesteps available (out of 157)')
ax.set_ylabel('Negative log-likelihood (reconstruction error)')
plt.tight_layout()
plt.savefig('figures/figure4a.png', transparent=True, dpi=400)
