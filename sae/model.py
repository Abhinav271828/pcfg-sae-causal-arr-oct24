from typing import Optional
import torch
from .sparsemax_enc import SimpleLatentProto
from torch import nn

class SAE(nn.Module):
    def __init__(self,
                 input_dim : int,
                 hidden_dim : int,
                 pre_bias : bool,
                 k : Optional[int],
                 sparsemax : str,
                 norm : Optional[str]):
        """
        Args:
            input_dim: int, dimensionality of input data
            hidden_dim: int, dimensionality of latent space
            pre_bias: bool, whether to use the usual formulation (bias before encoder, tied to bias after decoder)
            k: int, number of non-zero values in the latent space; if None, no top-k selection is performed
            sparsemax: bool, whether to use sparsemax in the encoder
            norm: str, normalization method. values:
                - None: no normalization
                - 'input': normalize the input (outside the model)
                - 'dec': normalize the decoder columns
                - 'recon': normalize the reconstruction
                or any combination of the above separated by +
        """
        super(SAE, self).__init__()

        self.k = k
        self.norm = norm.split('+') if norm else []
        self.pre_bias = pre_bias

        if sparsemax:
            match sparsemax:
                case 'no-kds':
                    self.encoder = SimpleLatentProto(input_dim, hidden_dim, kds_encoder=False)
                case 'recon-dist':
                    self.encoder = SimpleLatentProto(input_dim, hidden_dim, kds_encoder=True, kds_mode='recon_dist')
                case'dist':
                    self.encoder = SimpleLatentProto(input_dim, hidden_dim, kds_encoder=True, kds_mode='dist')
        elif pre_bias:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU())
            self.bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False), nn.ReLU())

        if 'dec' in self.norm:
            self.decoder = nn.Parameter(torch.randn(hidden_dim, input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
    
    def forward(self, x):
        if self.pre_bias:
            x = x - self.bias
        latent = self.encoder(x)

        if self.k:
            values, indices = torch.topk(latent, self.k)
            latent = torch.zeros_like(latent)
            latent.scatter_(-1, indices, values)

        if 'dec' in self.norm:
            norm = torch.norm(self.decoder, p=2, dim=1) + 1e-8
            recon = torch.matmul(latent, self.decoder / norm.unsqueeze(1))
        else:
            recon = self.decoder(latent)

        if self.pre_bias:
            recon = recon + self.bias
        
        if 'recon' in self.norm:
            norm = torch.norm(recon, p=2, dim=1) + 1e-8
            recon = recon / norm.unsqueeze(1)

        return latent, recon
