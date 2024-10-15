import numpy as np


def define_prior(prior_size: int, alpha: float = 7e-2, prior_type: str = 'dirichlet'):
    """
    Generate a sparse prior distribution.

    Args:
        prior_size (int): The size of the prior distribution.
        alpha (float, optional): The concentration parameter for the Dirichlet distribution. Default is 0.1.

    Returns:
        numpy.ndarray: A sparse prior distribution.
    """
    if prior_type == 'dirichlet':
        x = np.random.dirichlet(np.repeat(alpha, prior_size), size=1)

    elif prior_type == 'zipfian':
        x = 1 / (np.arange(1, prior_size+1) ** alpha)

    elif prior_type == 'uniform':
        x = np.ones(prior_size)

    elif prior_type == 'structured_zeros':
        x = np.zeros(prior_size)
        x[:int(prior_size * alpha)] = 1

    else:
        raise ValueError(f"Invalid prior type: {prior_type}")

    x = np.random.permutation(x)
    return (x / x.sum()).squeeze()