import hydra 
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from model import GPT
from dgp import get_dataloader
from datasets import *
from evals import grammar_evals

from utils import init_wandb, set_seed, save_config, open_log, cleanup
from utils import sanity_checks, configure_optimizers, update_cosine_warmup_lr
from utils import save_model, move_to_device, log_train, log_eval


@hydra.main(config_path="./config/train", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="influence_on_emergence")
    set_seed(cfg.seed)
    save_config(cfg)
    fp = open_log(cfg)
    device = cfg.device if torch.cuda.is_available() else 'cpu'

    if cfg.data.language == 'random':
        dataloader = torch.utils.data.DataLoader(
            RandomHierarchyModel(
                num_features=cfg.data.config.num_features,
                num_classes=1,
                num_synonyms=cfg.data.config.num_synonyms,
                num_layers=cfg.data.config.num_layers,
                input_format='long'),
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers)
    else:
        dataloader = get_dataloader(
            language=cfg.data.language,
            config=cfg.data.config,
            alpha=cfg.data.alpha,
            prior_type=cfg.data.prior_type,
            num_iters=cfg.data.num_iters * cfg.data.batch_size,
            max_sample_length=cfg.data.max_sample_length,
            seed=cfg.seed,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
        )

    # Check if model is compatible with data
    if cfg.data.language != 'random':
        sanity_checks(cfg, dataloader.dataset.max_sample_length)

    # Define model
    if cfg.data.language == 'random':
        model = GPT(cfg.model, dataloader.dataset.num_features + 2)
    else:
        model = GPT(cfg.model, dataloader.dataset.PCFG.vocab_size)
    model.to(device)
    if cfg.model.compile:
        model = torch.compile(model)
    print("number of parameters: %.2fM" % (model.get_num_params()/1e6,))

    # Optimizer
    optimizer = configure_optimizers(model, cfg.optimizer)

    # Train
    train(cfg, model, dataloader, optimizer, device)

    # Close wandb and log file
    cleanup(cfg, fp)


def train(cfg, model, dataloader, optimizer, device):
    """
    Training function
    """
    # Set model to train mode
    model.train()

    # Set to True to save the grammar underlying the dataset
    save_grammar = True

    # Data type (bf16 for efficiency)
    dt = torch.bfloat16 if cfg.bf16 else torch.float32

    # Configuration
    total_steps = len(dataloader) * cfg.epochs

    # Initialize losses (NOTE: Records losses for each task token as well as the total loss)
    train_loss = {'total': []}
    for k in ['Task0']:
        train_loss[k] = []
    lr, it, save_tables = 0.0, 0, 0
    print("Total training steps: ", total_steps)
    print("Learning rate warmup steps: ", cfg.optimizer.warmup_iters)

    # Save initial model
    results_dir = save_model(cfg, model, optimizer, it)

    # Load pretrained model if desired
    if cfg.model.use_pretrained:
        model.load_state_dict(torch.load(cfg.model.pretrain_dir)['net'])
        optimizer.load_state_dict(torch.load(cfg.model.pretrain_dir)['optimizer'])

    # Training loop
    for e in range(cfg.epochs):
        for sequences, seq_lengths in tqdm(dataloader, desc=f'Epoch {e}'):
            if it > 7e4: # Training destabilizes after a certain point when the loss is too low, so we break
                save_model(cfg, model, optimizer, it)
                break

            # Split sequences into inputs and labels
            # (B, L) -> (B, L-1), (B, L-1)
            B = sequences.size(0)
            inputs, labels = move_to_device([sequences[:,:-1], sequences[:,1:]], device)
            # CORR: This overwrites the inputs as well, so indices get out of range. See loss computation
            #labels[labels == dataloader.dataset.pad_token_id] = -100  # Mask padding

            # Task tokens
            # NOTE: Only free generation is implemented for now. Nothing needs to change here when other tasks are implemented though.
            if cfg.data.language == 'random':
                samples_per_task = {'Task0': torch.ones(B, dtype=torch.bool)}
            else:
                samples_per_task = {
                    k: inputs[:, 1] == dataloader.dataset.task_token_idx[k]
                    for k in dataloader.dataset.task_token_idx
                }

            # Sequence statistics to log as well
            train_lengths = {
                'max': seq_lengths.max().item(), 
                'min': seq_lengths.min().item(), 
                'mean': seq_lengths.mean().item()
                }

            # Log train metrics
            if it % cfg.log.log_interval == 0:
                train_loss = log_train(it, cfg.deploy, lr, train_loss, train_lengths)

            # Evals
            if it % cfg.log.eval_interval == 0:
                model.eval() # Set to eval mode

                # Grammaticality results
                grammar_results_dict = grammar_evals(
                    cfg=cfg, model=model,
                    template=dataloader.dataset.template if cfg.data.language != 'random' else None, 
                    grammar=dataloader.dataset.PCFG if cfg.data.language != 'random' else dataloader.dataset,
                    device=device,
                    ) if cfg.eval.grammar else (None, None)
                
                # Log eval metrics
                save_tables = log_eval(deploy=cfg.deploy, it=it, save_tables=save_tables, 
                            grammaticality_results=grammar_results_dict,
                            ) 
                
                model.train() # Set back to train mode

            # Update LR
            it, lr = update_cosine_warmup_lr(it, cfg.optimizer, optimizer, total_steps)

            # Compute loss
            optimizer.zero_grad(set_to_none=True) # Set gradients to None
            with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=dt): # Mixed precision
                logits = model(inputs)  # (B, L-1, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    # CORR: To allow us to avoid the replacement above.
                    ignore_index=dataloader.dataset.pad_token_id if cfg.data.language != 'random' else -100,
                    reduction='none'
                    ) # (B*L-1)
                loss = loss.reshape(B, -1).mean(dim=1) # Sum over sequence length
                
                # Decompose loss according to task tokens
                for k in ['Task0']:
                    train_loss[k].append(loss[samples_per_task[k]].mean().item())

                loss = loss.mean() # Average loss
                train_loss['total'].append(loss.item()) # Log loss

            # Update model
            loss.backward() # Compute gradients
            if cfg.optimizer.grad_clip > 0.0: # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            optimizer.step() # Update weights

            # Save model every few iterations
            if it % cfg.log.save_interval == 0:
                save_model(cfg, model, optimizer, it)
            if save_grammar:
                dataloader.dataset.save_grammar(results_dir)
                save_grammar = False # Save only once
        
        # Save after every epoch
        save_model(cfg, model, optimizer, it)


if __name__ == "__main__":
    main()
