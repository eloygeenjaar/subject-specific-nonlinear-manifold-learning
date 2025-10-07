import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


def embed(config, checkpoint_p):
    # setup data_loader instances
    train_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    valid_dataloader = config.init_obj('data', module_data, **{'split': 'valid', 'shuffle': False})
    test_dataloader = config.init_obj('data', module_data, **{'split': 'test', 'shuffle': False})

    extra_args = {
        "input_size": train_dataloader.dataset.data_size,
        "num_subjects": train_dataloader.dataset.num_subjects,
        "latent_dim": train_dataloader.dataset.latent_dim if config['run_type'] != 'classification' else test_dataloader.dataset.num_classes
    }

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, **extra_args)
    checkpoint = torch.load(checkpoint_p, weights_only=False)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # get function handles of loss and metrics
    num_subjects = train_dataloader.dataset.num_subjects
    
    va_targets = torch.empty((num_subjects, )).long()
    va_embeddings = torch.empty((num_subjects, valid_dataloader.dataset.num_timesteps, test_dataloader.dataset.latent_dim))
    for (_, (x, ix, t_ix, y)) in enumerate(valid_dataloader):
        x = x.to(device, non_blocking=True).float()
        ix = ix.long()
        t_ix = t_ix.long()
        y = y.long()
        with torch.no_grad():
            output = model(x, ix)
        if 'fBIRN' in config['data']['type']:
            va_targets[ix] = y
            va_embeddings[ix] = output['z'].detach().cpu().view(x.size(0), x.size(1), -1)
        else:
            va_embeddings[ix, t_ix] = output['z'].detach().cpu()
        
    print(va_embeddings.min(0), va_embeddings.max(0))
    np.save(f'embeddings/{config["data"]["type"]}{config["data"]["args"]["roi_name"]}_{config["name"]}_embeddings.npy', va_embeddings.detach().cpu().numpy())
    
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
    embed(config, checkpoint_p)
