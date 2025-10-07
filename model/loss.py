import torch
import torch.nn.functional as F
from torch import distributions as D


def cross_entropy(output, target):
    logits = output['logits']
    if logits.size(-1) > 1:
        return F.cross_entropy(logits, target.long())
    else:
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), target.float())

def mse(output, target):
    return F.mse_loss(output['x_hat'], output['x_orig'])

def elbo(output, target):
    dist = output['dist']
    kl_loss = output['lambda'] * 10 * D.kl.kl_divergence(dist, D.Normal(0., 1.)).sum(-1)
    nll_loss = D.Normal(output['x_hat'], 1.0).log_prob(output['x_orig']).sum(-1)
    return (kl_loss - nll_loss).mean()

def temporal_elbo(output, target):
    dist = output['dist']
    # Prior
    first_timestep = D.Normal(torch.cat((
        torch.zeros_like(dist.mean[:, 0]).unsqueeze(1),
        dist.mean[:, :-1]), dim=1), 1.0)
    kl_loss = output['lambda'] * 50 * D.kl.kl_divergence(dist, first_timestep).sum(-1).view(-1)
    nll_loss = D.Normal(output['x_hat'], 1.0).log_prob(output['x_orig']).sum(-1)
    return (kl_loss - nll_loss).mean()
