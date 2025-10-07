import torch
from torcheval.metrics import functional as TEF
from torch.nn import functional as F
from torch import distributions as D


def accuracy(output, target):
    logits = output['logits']
    with torch.no_grad():
        if logits.size(-1) > 1:
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
        else:
            probs = torch.sigmoid(logits).squeeze(-1)
            pred = (probs >= 0.5)
        correct = torch.sum(pred == target.squeeze(-1)).detach()
    return correct / target.size(0)

def auroc(output, target):
    logits = output['logits']
    if logits.size(-1) > 1:
        probs = F.softmax(logits, dim=1)
        num_classes = logits.size(-1)
        return TEF.multiclass_auroc(probs, target.long(), num_classes=num_classes, average='macro')
    else:
        probs = torch.sigmoid(logits).squeeze(-1)
        return TEF.binary_auroc(probs, target.long())

def auprc(output, target):
    logits = output['logits']
    if logits.size(-1) > 1:
        probs = F.softmax(logits, dim=1)
        num_classes = logits.size(-1)
        return TEF.multiclass_auprc(probs, target.long(), num_classes=num_classes, average='macro')
    else:
        probs = torch.sigmoid(logits).squeeze(-1)
        return TEF.binary_auprc(probs, target.long())
    
def mse(output, target):
    return F.mse_loss(output['x_hat'], output['x_orig'])

def kl_loss(output, target):
    return D.kl.kl_divergence(output['dist'], D.Normal(0., 1.)).mean()

def temporal_kl_loss(output, target):
    return output['lambda'] * D.kl.kl_divergence(
        D.Normal(output['dist'].mean[:-1], output['dist'].stddev[:-1]),
        D.Normal(output['dist'].mean[1:], output['dist'].stddev[1:])).mean()

def nll(output, target):
    return -D.Normal(output['x_hat'], 0.1).log_prob(output['x_orig']).mean()

def temporal_l2(output, target):
    means = output['dist'].mean
    return ((means[:, 1:] - means[:, :-1]).abs() / torch.max(means[:, 1:].abs(), dim=-1, keepdim=True)[0]).sum(-1).mean()

def temporal_corr(output, target):
    means = output['dist'].mean
    onemean = means[:, 1:].detach().clone()
    twomean = means[:, :-1].detach().clone()
    onemean -= onemean.mean(1, keepdim=True)
    onemean /= (onemean.std(1, keepdim=True) + 1E-9)
    twomean -= twomean.mean(1, keepdim=True)
    twomean /= (twomean.std(1, keepdim=True) + 1E-9)
    corr = torch.einsum('ijk,ijk->ik', onemean, twomean) / onemean.size(1)
    return corr.mean()

