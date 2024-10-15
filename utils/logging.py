import torch
import wandb
import numpy as np
import random
import os
import sys
import warnings
import yaml
from omegaconf import OmegaConf


# Sanity checks
def sanity_checks(cfg, max_sample_length):
    """
    Basic sanity checks for model configuration and data compatibility
    """

    # Check if vocabulary size and sequence length are compatible
    assert(cfg.model.context_size >= max_sample_length)
    assert(cfg.model.n_embd % cfg.model.n_head == 0)

    # Check if BF16 is supported
    if not torch.cuda.is_available():
        warnings.warn("WARNING: running on CPU", UserWarning)
    else:
        if not torch.cuda.is_bf16_supported():
            warnings.warn("WARNING: running without BF16", UserWarning)

        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise NotImplementedError("Flash Attention requires PyTorch >= 2.0")


# Seed
def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    # rng = np.random.default_rng(seed)
    # true_seed = int(rng.integers(2**30))

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Wandb and logging
def open_log(cfg):
    """
    Open log file and redirect stdout and stderr to it
    """
    print(cfg)
    os.makedirs('logs/' + cfg.tag, exist_ok=True)
    # CORR: log in stdout
    #if cfg.deploy:
        #fname = 'logs/' + cfg.tag + '/' + wandb.run.id + ".log"
        #fout = open(fname, "a", 1)
        #sys.stdout = fout
        #sys.stderr = fout
        #print(cfg)
        #return fout


def save_config(cfg):
    """
    Save configuration to file
    """
    results_dir = 'results/' + cfg.tag + "/" + wandb.run.id
    os.makedirs(results_dir, exist_ok=True)
    with open(results_dir + '/conf.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg), f)


def init_wandb(cfg, project_name):
    """
    Initialize wandb
    """
    if cfg.deploy:
        wandb.init(project=project_name)
        wandb.run.name = wandb.run.id
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg))


def cleanup(cfg, fp):
    """
    Close log file and wandb
    """
    if cfg.deploy:
        fp.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()


def log_train(it, deploy, lr, train_loss, train_lengths):
    """
    Log training information
    """
    if deploy and len(train_loss) > 0:
        wandb.log({
            "train": {k: np.mean(v) for k, v in train_loss.items()},
            "iteration": it,
            "lr": lr
            })

        for k, v in train_lengths.items():
            wandb.log({'train': {f'lengths/{k}': v}})

    print("train -- iter: %d, lr: %.6f, loss: %.4f" % (it, lr, np.mean(train_loss['total'])))
    train_loss = {k: [] for k in train_loss.keys()}
    return train_loss


def log_eval(deploy, it, save_tables, grammaticality_results):
    """
    Log eval information
    """

    if deploy:
        wandb.log({'eval': {'iteration': it}})

        # Grammaticality
        if grammaticality_results is not None:
            for key in grammaticality_results.keys():
                if key == 'failures':
                    continue

                elif key == 'validity':
                    wandb.log({'grammaticality': {'validity': grammaticality_results['validity']}})

                else:
                    for k, v in grammaticality_results[key].items():
                        wandb.log({'grammaticality': {f'{key} ({k})': v}})

    print("eval -- iter: %d" % it)

    return save_tables+1


# Save model
def save_model(cfg, net, optimizer, it):
    """
    Save model checkpoint
    """
    if cfg.deploy:
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter': it,
            'config': cfg,
        }
        fdir = 'results/' + cfg.tag + "/" + wandb.run.id
        os.makedirs(fdir, exist_ok=True)
        if cfg.log.save_multiple:
            fname = os.path.join(fdir, 'ckpt_' + str(it+1) + '.pt')
        else:
            fname = os.path.join(fdir, 'latest_ckpt.pt')
        torch.save(checkpoint, fname)
        return fdir

