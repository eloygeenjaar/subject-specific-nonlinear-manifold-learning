import torch
import torch.nn as nn
import torch.nn.functional as F
import model.modules as modules
import numpy as np
import pandas as pd
from torch import distributions as D
from base import BaseModel


class AutoEncoder(BaseModel):
    def __init__(self, input_size, layer_type, num_layers, hidden_dim, latent_dim, num_subjects, dropout):
        super().__init__()
        self.input_size = input_size
        self.layer_type = layer_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_subjects = num_subjects
        self.dropout = dropout

        self.in_lin = torch.compile(getattr(modules, f'{layer_type}Linear')(
            input_size, hidden_dim, bias=False, num_subjects=num_subjects, encoder=True))
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.out_lin = getattr(modules, f'{layer_type}Linear')(
            hidden_dim, input_size, bias=True, num_subjects=num_subjects, encoder=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.enc_mlp = modules.EncMLP(hidden_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.dec_mlp = modules.DecMLP(latent_dim, hidden_dim, hidden_dim, num_layers, dropout)

    def forward(self, x, ix):
        z = self.in_lin(x, ix)
        z = self.bn(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.enc_mlp(z)
        x_hat = self.dec_mlp(z)
        x_hat = self.out_lin(x_hat, ix)
        return {'x_orig': x, 'x_hat': x_hat, 'z': z}

    def encode(self, x, ix):
        x = self.in_lin(x, ix)
        x = self.act(x)
        x = self.dropout(x)
        z = self.enc_mlp(x)
        return z

    def decode(self, z, ix):
        z = self.dec_mlp(z)
        x_hat = self.out_lin(z, ix)
        return x_hat

    def decode_mean(self, z):
        z = self.dec_mlp(z)
        x_hat = self.out_lin.forward_mean(z)
        return x_hat

    def decode_s(self, z, s):
        z = self.dec_mlp(z)
        x_hat = self.out_lin.forward_s(z, s)
        return x_hat

class VAE(BaseModel):
    def __init__(self, input_size, layer_type, num_layers, hidden_dim, latent_dim, num_subjects, dropout):
        super().__init__()
        self.input_size = input_size
        self.layer_type = layer_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_subjects = num_subjects
        self.dropout = dropout

        self.in_lin = torch.compile(getattr(modules, f'{layer_type}Linear')(
            input_size, hidden_dim, bias=False, num_subjects=num_subjects, encoder=True))
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.out_lin = getattr(modules, f'{layer_type}Linear')(
            hidden_dim, input_size, bias=True, num_subjects=num_subjects, encoder=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.enc_mlp = modules.EncMLP(hidden_dim, hidden_dim, latent_dim * 2, num_layers, dropout)
        self.dec_mlp = modules.DecMLP(latent_dim, hidden_dim, hidden_dim, num_layers, dropout)

    def forward(self, x, ix):
        x_size = x.size()
        if len(x_size) > 2:
            x = x.view(x_size[0] * x_size[1], x_size[2])
            ix = ix.unsqueeze(1).repeat(1, x_size[1]).view(-1)
        z = self.in_lin(x, ix)
        z = self.bn(z)
        z = self.act(z)
        z = self.dropout(z)
        mu_sd = self.enc_mlp(z)
        if len(x_size) > 2:
            mu_sd = mu_sd.view(x_size[0], x_size[1], self.latent_dim*2)
        mu, logvar = torch.split(mu_sd, self.latent_dim, dim=-1)
        sd = torch.exp(0.5 * logvar).clamp(1E-5, 5)
        dist = D.Normal(mu, sd)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        if len(x_size) > 2:
            z = z.view(x_size[0] * x_size[1], self.latent_dim)
        x_hat = self.dec_mlp(z)
        x_hat = self.out_lin(x_hat, ix)
        return {'x_orig': x, 'x_hat': x_hat, 'z': z, 'dist': dist}

    def encode(self, x, ix):
        x = self.in_lin(x, ix)
        x = self.act(x)
        x = self.dropout(x)
        mu_sd = self.enc_mlp(x)
        mu, logvar = torch.split(mu_sd, self.latent_dim, dim=-1)
        return mu

    def decode(self, z, ix):
        z = self.dec_mlp(z)
        x_hat = self.out_lin(z, ix)
        return x_hat

    def decode_mean(self, z):
        z = self.dec_mlp(z)
        x_hat = self.out_lin.forward_mean(z)
        return x_hat

    def decode_s(self, z, s):
        z = self.dec_mlp(z)
        x_hat = self.out_lin.forward_s(z, s)
        return x_hat

class Classifier(BaseModel):
    def __init__(self, input_size, layer_type, num_layers, hidden_dim, latent_dim, num_subjects, dropout):
        super().__init__()
        self.input_size = input_size
        self.layer_type = layer_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim if latent_dim > 2 else 1
        self.num_subjects = num_subjects
        self.dropout = dropout

        self.in_lin = torch.compile(getattr(modules, f'{layer_type}Linear')(
            input_size, hidden_dim, bias=False, num_subjects=num_subjects, encoder=True))
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.enc_mlp = modules.EncMLP(hidden_dim, hidden_dim, hidden_dim, num_layers, dropout)
        self.dropout_last = nn.Dropout(0.5)
        self.lin = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, ix):
        x = self.in_lin(x, ix)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.act(self.bn2(self.enc_mlp(x)))
        x = self.dropout_last(self.lin(x))
        return {
            'logits': x,
        }
