import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
seeds = [42, 1337, 9999, 1212]
subjects = range(15)
roi_dict = {
    'Forrest': (['AUD', 'DMN', 'CC', 'WholeBrain'], ['Auditory', 'Precuneus', 'Cognitive control', 'Whole brain']),
    'Sherlock': (['PMC', 'EV', 'AUD', 'WholeBrain'], ['Posteromedial Cortex', 'Early Visual', 'Auditory', 'Whole brain'])
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

columns = ['model', 'dataset', 'roi', 'seed', 'label', 'result']
palette = {
        'Group': '#B8E3FF',
        'Decomposed': '#8776B3',
        'Subject': '#C8B0EE',
        'MRMD-AE': '#83BAF2',
        # #83BAF2- blue
    }

mrmdae_results_dict = {
    'EV': {'IndoorOutdoor': 0.7760, 'MusicPresent': 0.7170},
    'AUD': {'IndoorOutdoor': 0.7523, 'MusicPresent': 0.7128},
    'PMC': {'IndoorOutdoor': 0.7318, 'MusicPresent': 0.6845},
    'WholeBrain': {'IndoorOutdoor': 0., 'MusicPresent': 0.}
}

i = 0
for (d_ix, dataset) in enumerate(datasets):
    rois, roi_names = roi_dict[dataset]
    labels = label_dict[dataset]
    label_names = label_names_dict[dataset]
    fig, axs = plt.subplots(1, len(labels), figsize=(20, 6))
    models = model_dict[dataset]
    print(models)
    results_df = pd.DataFrame(
        np.empty((len(models) * len(rois) * len(seeds) * len(labels)
                , len(columns))),
        columns=columns
    )
    
    for (r_ix, roi) in enumerate(rois):
        for (m_ix, model) in enumerate(models):
            if model == 'MRMD-AE':
                results_arr = np.zeros((len(seeds), 15, len(labels)))
                for (label_ix, label) in enumerate(labels):
                    results_arr[:, :, label_ix] = mrmdae_results_dict[roi][label]
            else:
                p = Path('saved/autoencoder/models') / f'{dataset}{roi}' / f'{model}-AutoEncoder' / 'results.npy'
                if p.is_file():
                    results = np.load(p, allow_pickle=True).item()
                    results_arr = np.empty((len(seeds), 15, len(labels)))
                    for (s, seed) in enumerate(seeds):
                        for subject in subjects:
                            for (label_ix, label) in enumerate(labels):
                                results_arr[s, subject, label_ix] = np.mean(results[seed][subject][label][:, 0])
                else:
                    results_arr = np.zeros((len(seeds), 15, len(labels)))
                
            results_arr = np.mean(results_arr, axis=1)
            print(results_arr.shape)
            for (s, seed) in enumerate(seeds):
                for (label_ix, label) in enumerate(labels):
                    print(model, s, label_ix, labels, results_arr[s, label_ix])
                    results_df.loc[i, 'model'] = model
                    results_df.loc[i, 'dataset'] = dataset
                    results_df.loc[i, 'roi'] = roi
                    results_df.loc[i, 'seed'] = seed
                    results_df.loc[i, 'result'] = results_arr[s, label_ix]
                    results_df.loc[i, 'label'] = label
                    i += 1

    results_df['result'] = results_df['result'] * 100

    # https://github.com/webermarcolivier/statannot/blob/master/example/example.ipynb     
    x = "roi"
    y = "result"
    hue = "model"
    for (i, label) in enumerate(labels):
        pairs = []
        for roi in rois:
            if roi != 'WholeBrain':
                for model_name in ['Subject', 'Decomposed']:
                    pairs.append(((roi, 'Group'), (roi, model_name)))
                    if dataset == 'Sherlock':
                        pairs.append(((roi, 'MRMD-AE'), (roi, model_name)))
            else:
                pairs.append(((roi, 'Group'), (roi, 'Decomposed')))
        hue_order = models
        results = results_df.loc[results_df['label'] == label].copy()

        bot = results.loc[results['result'] != 0, 'result'].min()
        top = results['result'].max()
        axs[i].set_ylim((bot - 5, min(100, top + 5)))
        ax = sns.boxplot(data=results, x=x, y=y, hue=hue, hue_order=hue_order, ax=axs[i], palette=palette)        
        annot = Annotator(ax, pairs, data=results, x=x, y=y, order=rois, hue=hue, hue_order=hue_order)
        annot.configure(test='t-test_ind', text_format='star', loc='inside')
        annot.apply_test()
        annot.annotate()
        
        #bot, top = ax.get_ylim()
        
        ax = sns.barplot(data=results, x=x, y=y, hue=hue, hue_order=hue_order, ax=axs[i], palette=palette, alpha=0.5)
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_title(label_names[i])
        axs[0].set_ylabel('Classification accuracy (%)')
        axs[i].set_xticklabels(roi_names)
        axs[i].get_legend().remove()
        axs[i].tick_params(axis='x', labelrotation=20)
    
    fig.suptitle(suptitle_dict[dataset])
    plt.tight_layout()
    if d_ix == 0:
        plt.savefig('figures/figure2a.png', transparent=True, dpi=400)
    else:
        plt.savefig('figures/figure2b.png', transparent=True, dpi=400)
    plt.clf()
    plt.close(fig)
