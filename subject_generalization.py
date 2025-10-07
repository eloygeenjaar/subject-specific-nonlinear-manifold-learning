import copy
import argparse
import torch
import pandas as pd
import nibabel as nb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch
import model.loss as module_loss
from parse_config import ConfigParser
from sklearn.decomposition import PCA
from utils.util import SubsetDataset
from base.base_data_loader import BaseDataLoader

# From: https://stackoverflow.com/questions/35668219/how-to-set-up-a-custom-font-with-custom-path-to-matplotlib-global-font
fe = fm.FontEntry(
    fname='/data/users1/egeenjaar/fonts/montserrat/static/Montserrat-SemiBold.ttf',
    name='Montserrat')
fm.fontManager.ttflist.append(fe)

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = fe.name

def generalize(config, checkpoint_p):

    # setup data_loader instances
    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    test_dataloader = config.init_obj('data', module_data, **{'split': 'test', 'shuffle': False})

    print(len(valid_dataloader.dataset), len(test_dataloader.dataset))

    extra_args = {
        "input_size": valid_dataloader.dataset.data_size,
        "num_subjects": valid_dataloader.dataset.num_subjects,
        "latent_dim": valid_dataloader.dataset.latent_dim if config['run_type'] != 'classifier' else test_dataloader.dataset.num_classes
    }
    
    seed = config['seed']
    print(f'---- THIS IS SEED: {seed}')
    group_config = copy.deepcopy(config)
    group_config.update_model('Decomposed', 'Group')
    group_experiment_path = group_config.save_dir.parent
    group_config_p = group_experiment_path / f'{seed}-config.json'
    group_config = group_config.load_config(group_config_p)
    group_checkpoint_p = group_experiment_path / f'{seed}-best.pth'
    
    group_model = group_config.init_obj('arch', module_arch, **extra_args)
    group_checkpoint = torch.load(group_checkpoint_p)
    group_state_dict = group_checkpoint['state_dict']
    group_model.load_state_dict(group_state_dict)
    
    model = config.init_obj('arch', module_arch, **extra_args)
    checkpoint = torch.load(checkpoint_p)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    
    train_subjects = valid_dataloader.dataset.subjects
    train_subjects = np.asarray([s.name.zfill(12) for s in train_subjects])
    test_subjects = test_dataloader.dataset.subjects
    test_subjects = np.asarray([s.name.zfill(12) for s in test_subjects])
    nosubjects = np.array(['002058157217', '000855525613', '000967185140', '001007859987', '000302844543', '000959007410', '000866489781', '000876583541', '000800485677', '000857570467'])
    train_mask = ~np.isin(train_subjects, nosubjects)
    test_mask = ~np.isin(test_subjects, nosubjects)
    
    train_subjects_masked = train_subjects[train_mask]
    test_subjects_masked = test_subjects[test_mask]
    
    assert np.isin(train_subjects, test_subjects).sum() == 0
    
    df = pd.read_excel('/data/qneuromark/Data/FBIRN/ZN_Neuromark/clinData03-20-2012-limited.xls', dtype={'SubjectID': str})
    df['SubjectID'] = df['SubjectID'].str.zfill(12)
    df.drop_duplicates('SubjectID', keep='last', inplace=True)
    df.set_index('SubjectID', inplace=True)
    print(df.loc[train_subjects_masked, 'Demographics_nDEMOG_DIAGNOSIS'].value_counts())
    print(df.loc[test_subjects_masked, 'Demographics_nDEMOG_DIAGNOSIS'].value_counts())
    
    subject_weights = torch.exp(model.out_lin.s).detach().cpu().numpy()
    print('Reinitializing subject weights')
    model.reinitialize_subject_weights(test_dataloader.dataset.num_subjects)
    pca = PCA(n_components=2)
    
    train_pca_weights = pca.fit_transform(subject_weights)
    train_sz_mask = (df.loc[train_subjects[train_mask], 'Demographics_nDEMOG_DIAGNOSIS'] == 2)
    train_c_mask = (df.loc[train_subjects[train_mask], 'Demographics_nDEMOG_DIAGNOSIS'] == 1)
    
    test_sz_mask = (df.loc[test_subjects[test_mask], 'Demographics_nDEMOG_DIAGNOSIS'] == 2)
    test_c_mask = (df.loc[test_subjects[test_mask], 'Demographics_nDEMOG_DIAGNOSIS'] == 1)

    num_timesteps = test_dataloader.dataset.num_timesteps
    subsets = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    #subsets = [0.2]
    for subset in subsets:
        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch, **extra_args)
        checkpoint = torch.load(checkpoint_p)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        print('Reinitializing subject weights')
        model.reinitialize_subject_weights(test_dataloader.dataset.num_subjects)

        # prepare for (multi-device) GPU training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.train()
        
        # Obtain subset dataset
        #print(f'Subset: {subset}')
        subset_dataset = SubsetDataset(test_dataloader.dataset, subset)
        subset_timesteps = subset_dataset.num_timesteps
        batch_size = config['data']['args']['batch_size']
        test_dataloader = BaseDataLoader(subset_dataset, batch_size, shuffle=True, num_workers=5, use_sampler=False)
        print(f'Subset: {subset}, Num timesteps: {subset_timesteps}')

        # Turn off all weights except the subject-specific one
        for (name, parameter) in model.named_parameters():
            if name not in ['in_lin._orig_mod.s', 'out_lin.s']:
                parameter.requires_grad = False
            print(name, parameter.requires_grad)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        print(f'Training, subset: {subset}, seed: {seed}')
        print((subset == 0.2) and (seed == 42))
        plot_ix = 0
        model = model.to(device)
        for epoch in range(min(int(10000 * subset), 2000)):
            model.train()
            for (i, (x, subject_ix, temporal_ix, y)) in enumerate(test_dataloader):
                optimizer.zero_grad(None)
                x = x[:, :subset_timesteps].to(device, non_blocking=True).float()
                subject_ix = subject_ix.to(device, non_blocking=True).long()
                y = y.to(device, non_blocking=True).long()
                output = model(x, subject_ix)  
                output['lambda'] = 1.
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                #if subset == 0.2 and i % 50 == 0 and seed == 42:
                #    new_weights = torch.exp(model.out_lin.s).detach().cpu().numpy()
                #    test_pca_weights = pca.transform(new_weights)
                #    
                #    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                #    ax.scatter(
                #        train_pca_weights[train_mask][train_sz_mask, 0],
                #        train_pca_weights[train_mask][train_sz_mask, 1], color='#8776B3', alpha=0.75)
                #    ax.scatter(
                #        train_pca_weights[train_mask][train_c_mask, 0],
                #        train_pca_weights[train_mask][train_c_mask, 1], color='#83BAF2', alpha=0.75)
                #    ax.scatter(
                #        test_pca_weights[test_mask][test_sz_mask, 0],
                #        test_pca_weights[test_mask][test_sz_mask, 1], color='#8776B3', marker='X',
                #        s=4 * (plt.rcParams['lines.markersize'] ** 2))
                #    ax.scatter(
                #        test_pca_weights[test_mask][test_c_mask, 0],
                #        test_pca_weights[test_mask][test_c_mask, 1], color='#83BAF2', marker='X',
                #        s=4 * (plt.rcParams['lines.markersize'] ** 2))
                #    ax.set_xlabel('PCA component 1')
                #    ax.set_ylabel('PCA component 2')
                #    ax.set_title('PCA of subject-specific weights')
                #    fig.savefig(f'subject_weight_figures/{str(plot_ix).zfill(3)}.png')
                #    plt.clf()
                #    plt.close(fig)
                #    plot_ix += 1
        if (subset == 0.2) and (seed == 42):
            new_weights = torch.exp(model.out_lin.s).detach().cpu().numpy()
            test_pca_weights = pca.transform(new_weights)
            np.savez('results/subject_weights_results.npz',
                    train_mask=train_mask,
                    test_mask=test_mask,
                    train_pca_weights=train_pca_weights,
                    train_sz_mask=train_sz_mask,
                    train_c_mask=train_c_mask,
                    test_pca_weights=test_pca_weights,
                    test_sz_mask=test_sz_mask,
                    test_c_mask=test_c_mask)
        print(f'Evaluating decomposed model')
        model.eval()
        results_dict = {met.__name__: 0. for met in metrics}
        num_examples = 0
        for (i, (x, subject_ix, temporal_ix, y)) in enumerate(test_dataloader):
            x = x[:, subset_timesteps:].to(device, non_blocking=True).float()
            subject_ix = subject_ix.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).long()
            subject_ix = subject_ix
            if x.size(0) > 0:
                with torch.no_grad():
                    output = model(x, subject_ix)  
                for met in metrics:
                    results_dict[met.__name__] += met(output, y) * x.size(0)
                num_examples += x.size(0)
    
        print(f'--- Decomposed results: {subset} ---')
        for met in metrics:
            print(f'{met.__name__}: {results_dict[met.__name__] / num_examples}')  
        
        group_model = group_model.to(device)
        group_model.eval()
        print(f'Evaluating group model at epoch: {epoch}')
        results_dict = {met.__name__: 0. for met in metrics}
        num_examples = 0
        for (i, (x, subject_ix, temporal_ix, y)) in enumerate(test_dataloader):
            x = x[:, subset_timesteps:].to(device, non_blocking=True).float()
            subject_ix = subject_ix.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).long()
            subject_ix = subject_ix
            if x.size(0) > 0:
                with torch.no_grad():
                    output = group_model(x, subject_ix)  
                for met in metrics:
                    results_dict[met.__name__] += met(output, y) * x.size(0)
                num_examples += x.size(0)
    
        print(f'--- Group results: {subset} ---')
        for met in metrics:
            print(f'{met.__name__}: {results_dict[met.__name__] / num_examples}')  

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
    generalize(config, checkpoint_p)
