import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import model.model as module_arch
from parse_config import ConfigParser
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from data_loader.data_loaders import SubjectMoonsData
from statannotations.Annotator import Annotator


# From: https://stackoverflow.com/questions/35668219/how-to-set-up-a-custom-font-with-custom-path-to-matplotlib-global-font
fe = fm.FontEntry(
    fname='/data/users1/egeenjaar/fonts/montserrat/static/Montserrat-SemiBold.ttf',
    name='Montserrat')
fm.fontManager.ttflist.append(fe)

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = fe.name

plot_subject_results = True

palette = {
        'Group': '#B8E3FF',
        'Decomposed': '#8776B3',
        'Subject': '#C8B0EE'}

ds = SubjectMoonsData('train', None, None, noise_level=0.1)

x = ds.orig_data
y = ds.orig_y

data = ds.data_np
thetas = ds.thetas
data = data[:, np.argsort(thetas)]

fig, ax = plt.subplots(1, 1)
ax.scatter(x[y==0, 0], x[y==0, 1], s=20, c=palette['Group'])
ax.scatter(x[y==1, 0], x[y==1, 1], s=20, c=palette['Decomposed'])
print(ax.get_xlim(), ax.get_ylim())
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-1.45, 2.5])
ax.set_ylim([-0.85, 1.26])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.setp(ax.spines.values(), linewidth=3)
plt.savefig('figures_simulation/original_data.png', dpi=300, bbox_inches='tight')
plt.clf()
plt.close(fig)

num_subjects = [16, 32, 64, 128, 256, 512, 1024, 2048]
input_sizes = [64, 300, 100000]
hidden_size = 256

for (i_s, input_size) in enumerate(input_sizes):
    fig, ax = plt.subplots(1, 1, figsize=(3*1.5, 3))
    # Weight + bias
    g_ls = [input_size * hidden_size + hidden_size] * len(num_subjects)
    s_ls, d_ls = [], []
    for (n_ix, ns) in enumerate(num_subjects):
        # Subject weight + bias
        s_ls.append(
            (ns * input_size * hidden_size + ns * hidden_size) / g_ls[n_ix]
        )
        # Decomposed weight + bias
        d_ls.append(
            ((ns * hidden_size) +  # Subject-specific weights
            (input_size * hidden_size) + # U/VT
            (hidden_size * hidden_size) + # U/VT
            (hidden_size)) / g_ls[n_ix]
        )        

    ax.plot(num_subjects, d_ls, linewidth=5, alpha=0.9, color='#8776B3')
    ax.plot(num_subjects, s_ls, linewidth=5, alpha=0.9, color='#C8B0EE')
    if i_s < 2:
        ax.set_xticks([])
    else:
        ax.set_xlabel('Number of subjects')
    ax.set_ylabel(None)
    ax.set_yscale('log')
    plt.setp(ax.spines.values(), linewidth=4)
    plt.savefig(f'figures_simulation/scaling_{input_size}.png', dpi=300, bbox_inches='tight')

fig, axs = plt.subplots(8, 16, figsize=(20, 10))
for i in range(8):
    for j in range(16):
        subject_ix = i * 16 + j
        axs[i, j].scatter(data[y==0, subject_ix, 0], data[y==0, subject_ix, 1], s=5, c=palette['Group'])
        axs[i, j].scatter(data[y==1, subject_ix, 0], data[y==1, subject_ix, 1], s=5, c=palette['Decomposed'])
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].set_xticklabels([])
        axs[i, j].set_yticklabels([])
        axs[i, j].set_xlim([-2, 2])
        axs[i, j].set_ylim([-2, 2])
        plt.setp(axs[i, j].spines.values(), linewidth=4)
plt.savefig('figures_simulation/subjects.png', dpi=300, bbox_inches='tight')
plt.clf()
plt.close(fig)

if plot_subject_results:
    i=0
    seeds = [42, 1337, 9999, 1212]
    modules = ['Group', 'Subject', 'Decomposed']
    results_df = pd.DataFrame(
        np.zeros((3 * len(seeds) * 10, 3)),
        columns=['module', 'trial', 'result'])
    for (m_ix, module) in enumerate(modules):
        p = Path(f'saved/classification/models/SubjectMoonsWholeBrain/{module}-Classifier')
        for (s_ix, seed) in enumerate(seeds):
            checkpoint_p = p / f'{seed}-best.pth'
            config_p = p / f'{seed}-config.json'
            config = ConfigParser.load_config(config_p)
            model = config.init_obj('arch', module_arch, **{'input_size': 2, 'num_subjects': 128, 'latent_dim': 1})
            checkpoint = torch.load(checkpoint_p, weights_only=False)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            if module == 'Subject':
                subject_weights = model.in_lin._orig_mod.weight.detach().cpu().numpy()
                subject_weights = subject_weights.reshape(128, -1)
            elif module == 'Decomposed':
                subject_weights = model.in_lin._orig_mod.s.detach().cpu().numpy()
                print(subject_weights.shape)
            else:
                subject_weights = np.zeros((128, 2))

            param_grid = {'est__alpha': [0.1, 1.0, 10.0, 100.0, 1000], 'est__solver': ['auto', 'cholesky', 'sparse_cg']}

            pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('est', Ridge())
            ])

            kfold = KFold(n_splits=10, shuffle=True, random_state=42)
            for (fold_ix, (tr_index, te_index)) in enumerate(kfold.split(subject_weights)):

                if module == 'Group':
                    results_df.loc[i, 'module'] = module
                    results_df.loc[i, 'trial'] = f'{seed}_{fold_ix}'
                else:
                    tr_x = subject_weights[tr_index]
                    te_x = subject_weights[te_index]

                    tr_y = np.stack((np.sin(thetas[tr_index]), np.cos(thetas[tr_index])), axis=1)
                    te_y = np.stack((np.sin(thetas[te_index]), np.cos(thetas[te_index])), axis=1)

                    model = GridSearchCV(pipeline, param_grid, n_jobs=-1)
                    model.fit(tr_x, tr_y)
                    print(f'Best training score: {model.best_score_}, best hyperparameters: {model.best_params_}')

                    est = pipeline.set_params(**model.best_params_)
                    est.fit(tr_x, tr_y)
                    test_score = est.score(te_x, te_y)
                    results_df.loc[i, 'module'] = module
                    results_df.loc[i, 'trial'] = f'{seed}_{fold_ix}'
                    results_df.loc[i, 'result'] = test_score
                i += 1

    fig, ax = plt.subplots(1, 1)
    ax = sns.boxplot(data=results_df, x='module', y='result', hue='module', hue_order=modules, ax=ax, palette=palette)        
    annot = Annotator(ax, [('Group', 'Subject'), ('Group', 'Decomposed'), ('Subject', 'Decomposed')], data=results_df, x='module', y='result', order=modules)
    annot.configure(test='t-test_ind', text_format='star', loc='inside')
    annot.apply_test()
    annot.annotate()

    ax = sns.barplot(data=results_df, x='module', y='result', hue='module', hue_order=modules, ax=ax, palette=palette, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_ylabel('Test R-squared')
    ax.tick_params(axis='x', labelrotation=20)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.savefig('figures_simulation/subject_results.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

i=0
seeds = [42, 1337, 9999, 1212]
modules = ['Group', 'Subject', 'Decomposed']
results_df = pd.DataFrame(
    np.zeros((3 * len(seeds), 3)),
    columns=['module', 'trial', 'result'])
for (m_ix, module) in enumerate(modules):
    p = Path(f'saved/classification/models/SubjectMoonsWholeBrain/{module}-Classifier')
    results_dict = np.load(p / 'results.npy', allow_pickle=True).item()
    for (s_ix, seed) in enumerate(seeds):
        results_df.loc[i, 'module'] = module
        results_df.loc[i, 'trial'] = seed
        results_df.loc[i, 'result'] = results_dict[seed]['accuracy'] * 100
        i += 1

fig, ax = plt.subplots(1, 1)
ax = sns.boxplot(data=results_df, x='module', y='result', hue='module', hue_order=modules, ax=ax, palette=palette, legend=False)        
ax.set_ylim([0, 125])
annot = Annotator(ax, [('Group', 'Subject'), ('Group', 'Decomposed'), ('Subject', 'Decomposed')], data=results_df, x='module', y='result', order=modules)
annot.configure(test='t-test_ind', text_format='star', loc='inside')
annot.apply_test()
annot.annotate()
ax = sns.barplot(data=results_df, x='module', y='result', hue='module', hue_order=modules, ax=ax, palette=palette, alpha=0.5, legend=False)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([0, 20, 40, 60, 80, 100])
ax.set_xticks([])
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_ylabel('Test Classification Accuracy (%)')
ax.tick_params(axis='x', labelrotation=20)
plt.setp(ax.spines.values(), linewidth=3)
plt.savefig('figures_simulation/classification_results.png', dpi=300, bbox_inches='tight')
