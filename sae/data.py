import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from model import GPT
from dgp import PCFGDataset
from utils import DictToObj

import os
import pickle as pkl
import yaml
from tqdm import tqdm

class SAEData(Dataset):
    def __init__(self, model_dir : str, ckpt : str, layer_name : str, config : str, device : str = 'cuda'):
        """
        A class to generate data to train the SAE. It returns a batch of activations from a specified layer by making
        a forward pass through the GPT model. Note that this is online, and the activations are not cached.
        params:
            * model_dir  : path to the directory containing the model
            * ckpt       : name of the checkpoint file
            * layer_name : name of the layer to extract activations from. one of
                - "wte" [embedding layer]
                - "wpe" [positional encoding layer]
                - "attn{n}" [n-th layer's attention ; n = 0, 1]
                - "mlp{n}" [n-th layer's mlp; n = 0, 1]
                - "res{n}" [n-th layer; n = 0, 1]
                - "ln_f" [final layer-norm before the LM-head]
            * config   : name of the configuration file (a yaml file located in model_dir) for the DGP
        Each returned sample consists of a tensor of shape [total_tokens, emb_dim], where
            total_tokens is the actual length of the sequence (excluding the decorator, <eos>, and padding tokens), and
            emb_dim is the dimension of the embeddings.
        Batch collation is done by concatenating the samples along the first dimension. Note that this leads to a
        variable batch size, as the lengths of sequences can vary.
        """
        self.model_dir = model_dir
        self.ckpt = ckpt
        self.layer_name = layer_name
        self.device = device

        print("Loading model...")
        model_dict = torch.load(os.path.join(self.model_dir, self.ckpt), map_location=self.device)
        if config:
            with open(os.path.join(self.model_dir, config), 'r') as f:
                cfg = DictToObj(yaml.safe_load(f))
        else:
            cfg = model_dict['config']
        with open(os.path.join(self.model_dir, 'grammar/PCFG.pkl'), 'rb') as f:
            pcfg = pkl.load(f)
        self.model = GPT(cfg.model, pcfg.vocab_size).to(self.device)
        self.model.load_state_dict(model_dict['net'])
        self.model.eval()
        self.dataset = PCFGDataset(
            language=cfg.data.language,
            config=cfg.data.config,
            alpha=cfg.data.alpha,
            prior_type=cfg.data.prior_type,
            num_iters=cfg.data.num_iters * cfg.data.batch_size,
            max_sample_length=cfg.data.max_sample_length,
            seed=cfg.seed,
        )
        self.pad_token_id = self.dataset.pad_token_id
        print("Model loaded.")

        self.activation = None
        self.grad = None
        def hook_fwd(model, input, output):
            self.activation = output.detach().clone()
        self.hook_fwd = hook_fwd

        def intervention_hook(module, input, output):
            return self.intervention
        self.intervention_hook = intervention_hook

        match self.layer_name:
            case "wte":   self.module = self.model.transformer.wte
            case "wpe":   self.module = self.model.transformer.wpe
            case "attn0": self.module = self.model.transformer.h[0].attn
            case "mlp0":  self.module = self.model.transformer.h[0].mlp
            case "res0":  self.module = self.model.transformer.h[0]
            case "attn1": self.module = self.model.transformer.h[1].attn
            case "mlp1":  self.module = self.model.transformer.h[1].mlp
            case "res1":  self.module = self.model.transformer.h[1]
            case "ln_f":  self.module = self.model.transformer.ln_f
        self.handle_fwd = self.module.register_forward_hook(hook_fwd)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sequence, length = self.dataset[idx]
        # [seq_len]

        logits = self.model(sequence.unsqueeze(0).to(self.device)) # self.activation : [seq_len, emb_dim]
        loss = F.cross_entropy(logits.squeeze(0)[:-1], sequence[1:])
        loss.backward(retain_graph=True)

        activation = self.activation.squeeze(0)[self.dataset.decorator_length:self.dataset.decorator_length + int(length)]
        # [length, emb_dim]
        logits = logits.squeeze(0)[self.dataset.decorator_length:self.dataset.decorator_length + int(length)]

        return activation, logits, sequence
    
    def intervene(self, sequence, activations):
        self.handle_fwd.remove()

        length = activations.shape[0]
        self.intervention = torch.cat([activations,
                                       torch.zeros(128 - length, 128).to(self.device)],
                                       dim=0).unsqueeze(0)
        self.handle_fwd = self.module.register_forward_hook(self.intervention_hook)

        logits = self.model(sequence.unsqueeze(0).to(self.device))
        logits = logits.squeeze(0)[self.dataset.decorator_length:self.dataset.decorator_length + int(length)]

        self.handle_fwd.remove()
        self.handle_fwd = self.module.register_forward_hook(self.hook_fwd)

        return logits


"""
GPT(
  (transformer): ModuleDict(
    (wte): Embedding(31, 128)
    (wpe): Embedding(256, 128)
    (h): ModuleList(
      (0-1): 2 x Block(
        (ln_1): LayerNorm()
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=128, out_features=384, bias=False)
          (c_proj): Linear(in_features=128, out_features=128, bias=False)
        )
        (ln_2): LayerNorm()
        (mlp): MLP(
          (c_fc): Linear(in_features=128, out_features=512, bias=False)
          (gelu): GELU(approximate='none')
          (c_proj): Linear(in_features=512, out_features=128, bias=False)
        )
      )
    )
    (ln_f): LayerNorm()
  )
  (LM_head): Linear(in_features=128, out_features=31, bias=False)
)
"""

