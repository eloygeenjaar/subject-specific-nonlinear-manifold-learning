import argparse
import torch
import pandas as pd
import nibabel as nb
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from nilearn.image import index_img
from nilearn import masking
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

comp_ix = [
    68, 52, 97, 98, 44,
    20, 55,
    2, 8, 1, 10, 26, 53, 65, 79, 71,
    15, 4, 61, 14, 11, 92, 19, 7, 76,
    67, 32, 42, 69, 60, 54, 62, 78, 83, 95, 87, 47, 80, 36, 66, 37, 82,
    31, 39, 22, 70, 16, 50, 93,
    12, 17, 3, 6]

def classify(config, checkpoint_p):

    # setup data_loader instances
    train_dataloader = config.init_obj('data', module_data, **{'split': 'train', 'shuffle': False})

    extra_args = {
        "input_size": train_dataloader.dataset.data_size,
        "num_subjects": len(train_dataloader.dataset.subjects),
        "latent_dim": train_dataloader.dataset.latent_dim
    }

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, **extra_args)
    checkpoint = torch.load(checkpoint_p)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    subject_paths = np.asarray(train_dataloader.dataset.subjects)
    subjects = np.asarray([s_path.name for s_path in train_dataloader.dataset.subjects])
    subject_weights = torch.exp(model.out_lin.s).detach().cpu().numpy()
    
    df = pd.read_csv('/data/users1/egeenjaar/local-global/data/ica_fbirn/info_df.csv', index_col=0)
    df['analysis_ID'] = df['analysis_ID'].astype(str)
    df['analysis_ID'] = df['analysis_ID'].str.zfill(12)
    df.set_index('analysis_ID', inplace=True)
    df.dropna(axis=0, subset=['path'], inplace=True)

    mask = np.isin(subjects, list(df.index.values))
    subject_weights = subject_weights[mask].copy()
    subjects_masked = subjects[mask]

    subject_paths = subject_paths[mask]
    df = df.loc[subjects_masked].copy()

    print(df.head(10))

    y = (df['sz'] == 2).values

    mask_img = nb.load('images/fbirn_mask.nii')
    ica_ls = []
    for (i, row) in df.iterrows():
        timecourse_path = row['path']
        spatial_ica_path = timecourse_path.replace('_timecourses_ica_s1_.nii', '_component_ica_s1_.nii')
        spatial_img = nb.load(spatial_ica_path)
        spatial_img = index_img(spatial_img, comp_ix)
        ica_ls.append(masking.apply_mask(spatial_img, mask_img))

    ica_x = np.stack(ica_ls, axis=0)
    ica_x = np.reshape(ica_x, (ica_x.shape[0], -1))
    pca = PCA(n_components=254)
    ica_x = pca.fit_transform(ica_x)

    kfolds = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    folds_sw, folds_ica = [], []
    for (train_index, test_index) in kfolds.split(subject_weights, y):
        svm = SVC(kernel='rbf')
        svm.fit(subject_weights[train_index], y[train_index])
        folds_sw.append(svm.score(subject_weights[test_index], y[test_index]))
        print('sw', folds_sw[-1])
        del svm
        svm = SVC(kernel='rbf')
        svm.fit(ica_x[train_index], y[train_index])
        folds_ica.append(svm.score(ica_x[test_index], y[test_index]))  
        print('ica', folds_ica[-1])
        
    print('mean SW')
    print(np.mean(folds_sw))
    print('mean ICA')
    print(np.mean(folds_ica))

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    args = argparse.ArgumentParser(description='Joint single subject and group neural manifold learning')
    args.add_argument('-rc', '--run_config', default=None, type=str, required=True,
                      help='Run config file path (default: None)')
    args.add_argument('-mc', '--model_config', default=None, type=str, required=True,
                      help='Path to model config (default: None)')
    args.add_argument('-dc', '--data_config', default=None, type=str, required=True,
                      help='Path to data config (default: None)')
    args.add_argument('-hn', '--hyperparameter_index', default=0, type=int,
                      help='Hyperparamter index (default: 0)')
    args.add_argument('-sn', '--seed_index', default=0, type=int,
                      help='Seed index (default: 0)')
    args = args.parse_args()
    config, hyperparameter_permutations, seeds = ConfigParser.from_args(args)
    seed = seeds[args.seed_index]
    print(seed)
    experiment_path = config.save_dir.parent
    config_p = experiment_path / f'{seed}-config.json'
    config = config.load_config(config_p)
    checkpoint_p = experiment_path / f'{seed}-best.pth'
    # fix random seeds for reproducibility
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    classify(config, checkpoint_p)
