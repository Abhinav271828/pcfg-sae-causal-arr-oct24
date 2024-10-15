import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax


class SimpleLatentProto(nn.Module):
    """
    Defines a new "local" layer using sparsemax
    output is either Ax*sparsemax(transform(x)), or sparsemax(transform(x))
    where transform(x) can be 
        Ax, (when kds_encoder=False) 
        Ax-\lambda [|x-a|^2 , ...], (kds_encoder=True, kds_mode='recon_dist') or 
        -\lambda [|x-a|^2 , ...] (kds_encoder=True, kds_mode='dist')
    """
    def __init__(self, in_dim, out_dim, fixed_lambda_value=None, random_prototypes=False, \
                 neuron_wise_lambda=False, lambda_positive=False, normalize_weight=True, weight_init=None,\
                    solu_mode=False, kds_encoder=False, kds_mode='recon_dist'):
        """
        in_dim, out_dim: (int) input and output dimensions respectively

        fixed_lambda_value: (float, or None) if given, lambda is a fixed parameter. If None (default), lambda is a trainable parameter
        
        random_prototypes: (bool) If True, initialize prototypes randomly and freeze them (do not train)

        neuron_wise_lambda: (bool) if True, use a different value of lambda for each prototype

        lambda_positive: (bool) If True, force lambda to be positive (lambda**2 is used in the layer update)

        normalize_weight: (bool) If True, prototypes are normalized to lie on the unit hypersphere

        weight_init: (torch.tensor or None) If given, initialize prototypes with these weights. Default None (random initialization on the unit hypersphere)

        solu_mode: (bool) If True, output is Ax*sparsemax(transform(x)) (transform depends on other parameters). If False, output is sparsemax(transform(x))

        kds_encoder: (bool) If True, transform(x)=Ax-\lambda [|x-a|^2 , ...]

        kds_mode: (str) One of 'recon_dist', and 'dist'
            If 'recon_dist': transform(x)= Ax-\lambda [|x-a|^2 , ...]
            If 'dist': transform(x) = \lambda [|x-a|^2 , ...]
        """
        super().__init__()
        self.kds_encoder = kds_encoder
        self.kds_mode = kds_mode
        if weight_init is None:
            A = torch.randn((out_dim, in_dim))
            A = A/torch.norm(A, dim=-1, keepdim=True) #init wts to norm 1, irresp. of normalize_weight
        else:
            A = weight_init #initialization
        self.normalize_weight = normalize_weight
        self.lambda_positive = lambda_positive
        self.solu_mode = solu_mode #if true, returns Ax*sparsemax(Ax); else sparsemax(Ax)
        if not random_prototypes:
            self.weight = nn.Parameter(A)
        else:
            self.weight = nn.Parameter(A, requires_grad=False) #fixed parameter, cannot be trained
        if fixed_lambda_value is not None:
            self.lambd = fixed_lambda_value #not a trainable parameter
        else:
            if not neuron_wise_lambda:
                lambd_length = 1
            else:
                lambd_length = out_dim
            lambd = torch.square(torch.randn((lambd_length))) #random (positive) initialization
            self.lambd = nn.Parameter(lambd) #trainable parameter
            
            
    def forward(self, x_input):
        """
        x_input has shape (batch_size, in_dim)
        
        """
        flat = (len(x_input.shape) == 3)
        if flat:
            bz = x_input.size(0)
            x_input = x_input.flatten(0, 1)

        sparsemax = Sparsemax()
        if self.lambda_positive:
            lambda_value = self.lambd**2
        else:
            lambda_value = self.lambd
        if self.normalize_weight:
            weight_n = F.normalize(self.weight, dim=-1)   
        else: 
            weight_n = self.weight

        x_input = F.normalize(x_input, dim=1) #normalize input
        if not self.kds_encoder:
            x = torch.matmul(x_input, weight_n.T)
            x_out = sparsemax(lambda_value*x)
        else: #use the kds encoder
            if self.kds_mode=="recon_dist":
                x = torch.matmul(x_input, weight_n.T) - \
                    lambda_value*torch.square(torch.norm(x_input.unsqueeze(1)-weight_n.unsqueeze(0), dim=-1))
                x_out = sparsemax(x)
            elif self.kds_mode=='dist':
                x = -lambda_value*torch.square(torch.norm(x_input.unsqueeze(1)-weight_n.unsqueeze(0), dim=-1))
                x_out = sparsemax(x)
        #sparse probabilities by projecting x onto the probability simplex
        if self.solu_mode:
            x_lin = torch.matmul(x_input, weight_n.T)
            ret = x_lin*x_out
        else:
            ret = x_out

        if flat:
            ret = ret.reshape(bz, -1, ret.size(-1))
        return ret

