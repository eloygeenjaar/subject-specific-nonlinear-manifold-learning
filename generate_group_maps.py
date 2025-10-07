import argparse
import torch
import pandas as pd
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mc
import numpy as np
import model.model as module_arch
import data_loader.data_loaders as module_data
from parse_config import ConfigParser


def interpolate(config, checkpoint_p):

    extra_args = {
        "input_size": 60409,
        "num_subjects": 294,
        "latent_dim": 64
    }

    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, **extra_args)
    checkpoint = torch.load(checkpoint_p, weights_only=False)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    subject_weights = model.out_lin.s.exp().detach()

    train_subjects = valid_dataloader.dataset.subjects
    print(f'Total number of subjects: {len(train_subjects)}')
    train_subjects = np.asarray([s.name.zfill(12) for s in train_subjects])
    
    nosubjects = np.array(['002058157217', '000855525613', '000967185140', '001007859987', '000302844543', '000959007410', '000866489781', '000876583541', '000800485677', '000857570467'])
    train_mask = ~np.isin(train_subjects, nosubjects)

    train_subjects_masked = train_subjects[train_mask]
    
    df = pd.read_excel('/data/qneuromark/Data/FBIRN/ZN_Neuromark/clinData03-20-2012-limited.xls', dtype={'SubjectID': str})
    df['SubjectID'] = df['SubjectID'].str.zfill(12)
    df.drop_duplicates('SubjectID', keep='last', inplace=True)
    df.set_index('SubjectID', inplace=True)
    print(df.loc[train_subjects_masked, 'Demographics_nDEMOG_DIAGNOSIS'].value_counts())
    
    df_cminds = pd.read_csv('/data/users1/egeenjaar/local-global/data/ica_fbirn/info_df.csv')
    df_cminds['name'] = df_cminds['name'].astype(str)
    df_cminds['name'] = df_cminds['name'].str.zfill(12)
    df_cminds.set_index('name', inplace=True)
    cminds_subjects = df_cminds.index.values.tolist()
    cminds_mask = np.isin(train_subjects[train_mask], cminds_subjects)
    print(cminds_mask.sum(), cminds_mask.shape, len(train_subjects[train_mask]), len(cminds_subjects))

    train_sz_mask = (df.loc[train_subjects[train_mask], 'Demographics_nDEMOG_DIAGNOSIS'] == 2).values.astype(bool)
    train_dose = df_cminds.loc[train_subjects[train_mask][cminds_mask], 'CPZ'].values
    train_cminds = df_cminds.loc[train_subjects[train_mask][cminds_mask], 'CMINDS_composite'].values
    train_c_mask = (df.loc[train_subjects[train_mask], 'Demographics_nDEMOG_DIAGNOSIS'] == 1).values.astype(bool)

    np.save('results/dose.npy', train_dose)
    np.save('results/cminds.npy', train_cminds)
    np.save('results/cminds_mask.npy', cminds_mask)

    subject_weights = subject_weights[train_mask]
    num_subjects = subject_weights.size(0)
    subject_weights_rep = subject_weights.unsqueeze(0).repeat(3, 1, 1).view(-1, subject_weights.size(-1))

    latent_dims = extra_args['latent_dim']
    embeddings = np.load('fBIRNembeddings.npy')
    print(embeddings.shape)

    min_embeddings = np.min(embeddings[train_mask], axis=(0, 1))
    max_embeddings = np.max(embeddings[train_mask], axis=(0, 1))

    latent_dim_ls = list(range(64))

    nodes = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0]
    colors = [(0., 0., 0.5), (0., 0., 1.), (0., 1., 1.), (1, 1, 1), (1, 1, 0), (1, 0, 0), (0.5, 0, 0)]
    colors = [mc.to_hex(color) for color in colors]
    nodes = list(zip(nodes, colors))
    map_ls = []
    for ld_ix in latent_dim_ls:
        z = torch.zeros((3, num_subjects, latent_dims), device=device)
        minld = min_embeddings[ld_ix]
        maxld = max_embeddings[ld_ix]
        print(minld.shape, min_embeddings.shape)
        z[0, :, ld_ix] = torch.Tensor([minld]).float().to(device)
        z[2, :, ld_ix] = torch.Tensor([maxld]).float().to(device)
        print(minld, maxld)
        z = z.view(-1, latent_dims)
        with torch.no_grad():
            x_var = model.decode_s(
                z, subject_weights_rep)
            x_var = x_var.view(3, num_subjects, -1)
            x_var = torch.stack((x_var[0] - x_var[1], x_var[2] - x_var[1]), dim=0)
            x_var = x_var[x_var.abs().sum(dim=(1, 2)).argmax()]

            if x_var.mean(0).abs().max() >= 0.1:
                map_ls.append(x_var.detach().cpu())
    
    np.save('results/group_maps.npy', np.stack(map_ls, axis=1))
    np.save('results/group_maps_sz.npy', train_sz_mask)

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
    interpolate(config, checkpoint_p)
