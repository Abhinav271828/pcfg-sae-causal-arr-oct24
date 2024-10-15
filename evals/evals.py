import torch
import numpy as np

def grammar_evals(cfg, model, template, grammar, device):
    """
    Evaluate the model on grammaticality.

    Args:
        cfg (dict): Configuration dictionary.
        model (torch.nn.Module): Model to evaluate.
        template (torch.Tensor): Template to generate samples from.
        grammar (Grammar): Grammar object.
        device (torch.device): Device to run on.

    Returns:
        results_dict (dict): Results of the grammaticality evaluation.
    """
    model.eval()
    eval_bsize = 128

    with torch.no_grad():

        # Generate samples
        if template:
            inputs = template.repeat(eval_bsize, 1).to(device)
        else:
            inputs = torch.zeros(eval_bsize, 1, dtype=torch.long).to(device)
            inputs[:] = grammar.bos
        samples, per_token_logprobs = model.sample(
            inputs=inputs, 
            max_new_tokens=(cfg.data.max_sample_length) - 10 if template else \
                            grammar.max_sample_length + 1,
            retrieve_llhoods='tokens',
            )

        # Transfer to CPU and detokenize
        samples = samples.cpu().numpy()
        if cfg.data.language != 'random':
            samples = [grammar.detokenize_sentence(s).split('<eos>')[0] for s in samples]
        else:
            samples = [' '.join(f'T{x}' for x in s).split(f'T{grammar.eos}')[0] for s in samples]

        # Eval grammatical correctness
        results_grammaticality = {
            'validity': {'num': 0, 'satisfied': 0},
            'logprobs': {'max': 0, 'min': 0, 'mean': 0, 'distance': 0},
            'depth': {'max': 0, 'min': 0, 'mean': 0},
            'failures': None,
            'model_logprobs': {'max': 0, 'min': 0, 'mean': 0},
            'lengths': {'max': 0, 'min': 0, 'mean': 0}
            }

        logprobs, model_logprobs, depth, lengths = [], [], [], []
        failures = []
        for sid, s in enumerate(samples):
            results_grammaticality['validity']['num'] += 1
            grammaticality, n_tokens = grammar.check_grammaticality(s)
            failures.append(grammaticality[3])
            model_logprobs.append(per_token_logprobs[sid, :n_tokens].sum().item())
            lengths.append(float(n_tokens))

            if grammaticality[0]:
                results_grammaticality['validity']['satisfied'] += 1
                logprobs.append(grammaticality[1])
                depth.append(grammaticality[2])
            else:
                logprobs.append(-5.)
                depth.append(0.)

        logprobs, depth = torch.tensor(logprobs).float(), torch.tensor(depth).float()
        model_logprobs = torch.tensor(model_logprobs).float()
        lengths = torch.tensor(lengths).float()

        # Update results
        results_grammaticality['validity'] = (
            results_grammaticality['validity']['satisfied'] / 
            results_grammaticality['validity']['num']
            )

        results_grammaticality['logprobs']['max'] = logprobs.max().item()
        results_grammaticality['logprobs']['min'] = logprobs.min().item()
        results_grammaticality['logprobs']['mean'] = logprobs.mean().item()
        results_grammaticality['logprobs']['distance'] = (model_logprobs - logprobs).abs().mean().item()

        results_grammaticality['depth']['max'] = depth.max().item()
        results_grammaticality['depth']['min'] = depth.min().item()
        results_grammaticality['depth']['mean'] = depth.mean().item()

        results_grammaticality['model_logprobs']['max'] = model_logprobs.max().item()
        results_grammaticality['model_logprobs']['min'] = model_logprobs.min().item()
        results_grammaticality['model_logprobs']['mean'] = model_logprobs.mean().item()

        results_grammaticality['lengths']['max'] = lengths.max().item()
        results_grammaticality['lengths']['min'] = lengths.min().item()
        results_grammaticality['lengths']['mean'] = lengths.mean().item()

        results_grammaticality['failures'] = failures

        return results_grammaticality