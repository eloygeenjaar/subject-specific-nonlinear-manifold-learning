import torch
from torch import nn
from torch import jit
from torch.nn import functional as F
from torch.nn.utils import parametrize


class GroupLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool, num_subjects=None, encoder=None):
        super().__init__()
        self.lin = nn.Linear(input_size, output_size, bias)
        # The gain is roughly the same for GELUs as ReLUs
        nn.init.xavier_uniform_(self.lin.weight, gain=nn.init.calculate_gain('relu'))
        if bias:
            nn.init.constant_(self.lin.bias, 0.)

    def forward(self, x: torch.Tensor, ixs: torch.Tensor) -> torch.Tensor:
        return self.lin(x)

    def transpose_forward(self, x, ix):
        return F.linear(x, self.lin.weight.T)

class SubjectLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool, num_subjects=None, encoder=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_subjects, input_size, output_size))
        for i in range(num_subjects):
            # The gain is roughly the same for GELUs as ReLUs
            nn.init.xavier_uniform_(self.weight[i], gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.zeros(num_subjects, output_size), requires_grad=bias)

    def forward(self, x: torch.Tensor, ixs: torch.Tensor) -> torch.Tensor:
        return torch.einsum('si,sij->sj', x, self.weight[ixs]) + self.bias[ixs]

class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, X):
        return F.normalize(X, dim=self.dim)

class DecomposedLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool, num_subjects=None, encoder=None):
        super().__init__()
        self.encoder = encoder
        # W = (U, S, V^T), the T is the transpose
        if encoder:
            self.s = nn.Parameter(
                torch.zeros((num_subjects, output_size)))
            print(f'assigned s: {self.s.numel()}')
            self.U = nn.Linear(input_size, output_size, bias=False)
            print(f'Assigned U: {self.U.weight.numel()}')
            # The gain is roughly the same for GELUs as ReLUs
            nn.init.xavier_uniform_(self.U.weight, gain=nn.init.calculate_gain('relu'))
            #self.U = nn.utils.parametrizations.orthogonal(self.U, orthogonal_map='householder')
            self.U = nn.utils.parametrize.register_parametrization(self.U, 'weight', Normalize(dim=-1))
            self.Vt = nn.Linear(output_size, output_size, bias=bias)
            print(f'Vt initialized: {self.Vt.weight.numel()}')
            # The gain is roughly the same for GELUs as ReLUs
            nn.init.xavier_uniform_(self.Vt.weight, gain=nn.init.calculate_gain('relu'))
            self.Vt = nn.utils.parametrizations.orthogonal(self.Vt)
            self.Vt = nn.utils.parametrize.register_parametrization(self.Vt, 'weight', Normalize(dim=-2))
        else:
            self.s = nn.Parameter(
                torch.zeros((num_subjects, input_size)))
            self.U = nn.Linear(input_size, input_size, bias=False)
            # The gain is roughly the same for GELUs as ReLUs
            nn.init.xavier_uniform_(self.U.weight, gain=nn.init.calculate_gain('relu'))
            self.U = nn.utils.parametrizations.orthogonal(self.U)
            self.U = nn.utils.parametrize.register_parametrization(self.U, 'weight', Normalize(dim=-1))
            self.Vt = nn.Linear(input_size, output_size, bias=bias)
            # The gain is roughly the same for GELUs as ReLUs
            nn.init.xavier_uniform_(self.Vt.weight, gain=nn.init.calculate_gain('relu'))
            #self.Vt = nn.utils.parametrizations.orthogonal(self.Vt, orthogonal_map='householder')
            self.Vt = nn.utils.parametrize.register_parametrization(self.Vt, 'weight', Normalize(dim=-2))

    def forward(self, x: torch.Tensor, ixs: torch.Tensor) -> torch.Tensor:
        s = torch.exp(self.s[ixs])
        if self.encoder:
            x = self.U(x)
            x = x * s
            x = self.Vt(x)
        else:
            x = self.U(x)
            x = x * s
            x = self.Vt(x)
        return x

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.exp(self.s).mean(0).unsqueeze(0)
        if self.encoder:
            x = self.U(x)
            x = x * s
            x = self.Vt(x)
        else:
            x = self.U(x)
            x = x * s
            x = self.Vt(x)
        return x

    def forward_s(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.encoder:
            x = self.U(x)
            x = x * s
            x = self.Vt(x)
        else:
            x = self.U(x)
            x = x * s
            x = self.Vt(x)
        return x

    def reconstruct_W(self):
        print(self.U.weight.size(), self.Vt.weight.size())
        return (self.Vt.weight.unsqueeze(0) * torch.exp(self.s).unsqueeze(1)) @ self.U.weight

    def reinitialize_s(self, num_subjects):
        self.s = nn.Parameter(torch.randn(num_subjects, self.s.size(-1)) * 0.01)
        print(self.s.size())

    def transpose_forward(self, x, ix):
        s = torch.exp(self.s[ix])
        x = F.linear(x, self.Vt.weight.T)
        x = x * s
        x = F.linear(x, self.U.weight.T)
        return x
        
class EncMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, num_layers: int, dropout_val: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.act = nn.GELU
        # There is always a linear layer before this MLP
        # so it starts with an activation
        if self.num_layers == 0:
            self.layers.append(nn.Identity())
        # Multiple layers with a linear layer at the end
        else:
            in_size = input_size
            out_size = hidden_size
            for i in range(self.num_layers):
                if i < (self.num_layers - 1):
                    self.layers.extend([
                        nn.Linear(in_size, out_size, bias=False),
                        nn.BatchNorm1d(out_size),
                        self.act(),
                        nn.Dropout(dropout_val)
                        ]
                    )
                else:
                    self.layers.extend([
                        nn.Linear(in_size, output_size, bias=True)
                    ])
                in_size = hidden_size
                out_size = hidden_size

    def forward(self, x):
        _x = x.clone()
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                x = layer(x) + _x
                _x = x.clone()
        x = self.layers[-1](x)
        return x

class DecMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, num_layers: int, dropout_val: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.act = nn.GELU
        # There is always a linear layer before this MLP
        # so it starts with an activation
        if self.num_layers == 0:
            self.layers.append(nn.Identity())
        # Multiple layers with a linear layer at the end
        else:
            in_size = input_size
            out_size = hidden_size
            for i in range(self.num_layers):
                self.layers.extend([
                    nn.Linear(in_size, out_size, bias=False),
                    nn.BatchNorm1d(out_size),
                    self.act(),
                    nn.Dropout(dropout_val)
                    ]
                )
                in_size = hidden_size
                out_size = hidden_size

    def forward(self, x):
        x = self.layers[0](x)
        _x = x.clone()
        for layer in self.layers[1:]:
            if isinstance(layer, nn.Linear):
                x = layer(x) + _x
                _x = x.clone()
        return x
