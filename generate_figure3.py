import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from pathlib import Path
from statannotations.Annotator import Annotator

# From: https://stackoverflow.com/questions/35668219/how-to-set-up-a-custom-font-with-custom-path-to-matplotlib-global-font
fe = fm.FontEntry(
    fname='/data/users1/egeenjaar/fonts/montserrat/static/Montserrat-SemiBold.ttf',
    name='Montserrat')
fm.fontManager.ttflist.append(fe)

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = fe.name

datasets = ['Sherlock', 'Forrest']
models = ['Group', 'Subject', 'Decomposed']
seeds = [42, 1337, 9999, 1212]
subjects = range(15)
roi_dict = {
    'Forrest': (['AUD', 'DMN', 'CC', 'WholeBrain'], ['Auditory', 'Precuneus', 'Cognitive control', 'Whole brain']),
    'Sherlock': (['AUD', 'EV', 'PMC', 'WholeBrain'], ['Auditory', 'Early Visual', 'Posteromedial Cortex', 'Whole brain'])
}
model_dict = {
    'Forrest': ['Group', 'Subject', 'Decomposed'],
    'Sherlock': ['MRMD-AE', 'Group', 'Subject', 'Decomposed']
}
label_dict = {
    'Forrest': ['FoT_coded', 'IoE_coded', 'ToD_coded'],
    'Sherlock': ['MusicPresent', 'IndoorOutdoor']
}
label_names_dict = {
    'Forrest': ['Flow of Time', 'Interior or Exterior', 'Time of Day'],
    'Sherlock': ['Music presence', 'Indoor or Outdoor']
}
suptitle_dict = {
    'Forrest': 'Forrest Gump audio movie',
    'Sherlock': 'Sherlock Holmes audio/visual episode'
}

columns = ['model', 'dataset', 'roi', 'seed', 'label', 'result', 'result_normalized']

hue = 'model'
hue_order = ['Group', 'Subject', 'Decomposed']

palette = {
        'Group': '#B8E3FF',
        'Decomposed': '#8776B3',
        'Subject': '#C8B0EE',
        'MRMD-AE': '#83BAF2'
        #  #83BAF2- blue
        #3981B8 - dark blue
    }


i = 0
for (d_ix, dataset) in enumerate(datasets):
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    labels = label_dict[dataset]
    label_names = label_names_dict[dataset]
    rois, roi_names = roi_dict[dataset]
    results_df = pd.DataFrame(
        np.empty((len(models) * len(datasets) * len(rois) * len(seeds)
                , len(columns))),
        columns=columns
    )
    for (r_ix, roi) in enumerate(rois):
        for (m_ix, model) in enumerate(models):
            p = Path('saved/autoencoder/models') / f'{dataset}{roi}' / f'{model}-AutoEncoder' / 'results.npy'
            if p.is_file():
                results = np.load(p, allow_pickle=True).item()
                results_arr = np.empty((len(seeds)))
                for (s, seed) in enumerate(seeds):
                    results_arr[s] = float(results[seed]['mse'])
            else:
                results_arr = np.zeros((len(seeds)))
                
            if model == 'Group':
                print(roi, 'updated')
                results_arr_group = results_arr.copy()

            for (s, seed) in enumerate(seeds):
                print(results_df.loc[i, 'result'], results_arr[s])
                results_df.loc[i, 'model'] = model
                results_df.loc[i, 'dataset'] = dataset
                results_df.loc[i, 'roi'] = roi
                results_df.loc[i, 'seed'] = seed
                results_df.loc[i, 'result'] = results_arr[s]
                print(results_arr[s] / results_arr_group[s])
                results_df.loc[i, 'result_normalized'] = (1 - results_arr[s] / results_arr_group[s]) * 100
                results_df.loc[i, 'label'] = 'None'
                i += 1
            
    # https://github.com/webermarcolivier/statannot/blob/master/example/example.ipynb     
    x = "label"
    y = "result"
    y_new = 'result_normalized'
    hue = "model"
    for (i, roi) in enumerate(rois):
        if roi == 'WholeBrain':
            pairs = []
            pairs.append((('None', 'Group'), ('None', 'Decomposed')))
            hue_order = ['Group', 'Decomposed']
        else:
            pairs = []
            for model_name in ['Subject', 'Decomposed']:
                pairs.append((('None', 'Group'), ('None', model_name)))
            hue_order = models
        results = results_df.loc[results_df['roi'] == roi].copy()
        print(results)
        ax = sns.boxplot(data=results, x='label', y=y_new, hue=hue, hue_order=hue_order, ax=axs[i], palette=palette)
        annot = Annotator(ax, pairs, data=results, x=x, y=y, order=['None'], hue=hue, hue_order=hue_order)
        annot.configure(test='t-test_ind', text_format='star', loc='inside')
        annot.apply_test()
        annot.annotate()
        bot, top = ax.get_ylim()
        ax = sns.barplot(data=results, x='label', y=y_new, hue=hue, hue_order=hue_order, ax=axs[i], palette=palette, alpha=0.5)
        axs[i].set_ylim((bot, top))
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_title(roi_names[i])
        axs[0].set_ylabel('Reconstruction improvement (%)')
        axs[i].set_xticks([])
        axs[i].set_xticklabels('')
        axs[i].get_legend().remove()
        
    fig.suptitle(suptitle_dict[dataset])
    plt.tight_layout()
    if d_ix == 0:
        plt.savefig('figures/figure3a.png', transparent=True, dpi=400)
    else:
        plt.savefig('figures/figure3b.png', transparent=True, dpi=400)
    plt.clf()
    plt.close(fig)
