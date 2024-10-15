from itertools import *
from typing import Iterator, List, Tuple, Union
import warnings
import copy

import numpy as np
import random
import os
import pickle as pkl

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import nltk
Symbol = Union[str, nltk.grammar.Nonterminal]

from .utils import dec2bin, dec2base

def sample_rules( v, n, m, s, L, seed=42):
        """
        Sample random rules for a random hierarchy model.

        Args:
            v: The number of values each variable can take (vocabulary size, int).
            n: The number of classes (int).
            m: The number of synonymic lower-level representations (multiplicity, int).
            s: The size of lower-level representations (int).
            L: The number of levels in the hierarchy (int).
            seed: Seed for generating the rules.

        Returns:
            A dictionary containing the rules for each level of the hierarchy.
        """
        random.seed(seed)
        tuples = list(product(*[range(v) for _ in range(s)]))

        rules = {}
        rules[0] = torch.tensor(
                random.sample( tuples, n*m)
        ).reshape(n,m,-1)
        for i in range(1, L):
            rules[i] = torch.tensor(
                    random.sample( tuples, v*m)
            ).reshape(v,m,-1)

        return rules


def sample_data_from_generator_classes(g, y, rules, return_tree_structure=False):
    """
    Create data of the Random Hierarchy Model starting from its rules, a seed and a set of class labels.

    Args:
        g: A torch.Generator object.
        y: A tensor of size [batch_size, 1] containing the class labels.
        rules: A dictionary containing the rules for each level of the hierarchy.
        return_tree_structure: If True, return the tree structure of the hierarchy as a dictionary.

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    L = len(rules)  # Number of levels in the hierarchy

    labels = copy.deepcopy(y)

    if return_tree_structure:
        x_st = (
            {}
        )  # Initialize the dictionary to store the hidden variables
        x_st[0] = y
        for i in range(L):  # Loop over the levels of the hierarchy
            chosen_rule = torch.randint(
                low=0, high=rules[i].shape[1], size=x_st[i].shape, generator=g
            )  # Choose a random rule for each variable in the current level
            x_st[i + 1] = rules[i][x_st[i], chosen_rule].flatten(
                start_dim=1
            )  # Apply the chosen rule to each variable in the current level
        return x_st, labels
    else:
        x = y
        for i in range(L):
            chosen_rule = torch.randint(
                low=0, high=rules[i].shape[1], size=x.shape, generator=g
            )
            x = rules[i][x, chosen_rule].flatten(start_dim=1)
        return x, labels
    

def sample_with_replacement(train_size, test_size, seed_sample, rules):

    n = rules[0].shape[0]  # Number of classes

    if train_size == -1:
        warnings.warn(
            "Whole dataset (train_size=-1) not available with replacement! Using train_size=1e6.",
            RuntimeWarning,
        )
        train_size = 1000000

    g = torch.Generator()
    g.manual_seed(seed_sample)

    y = torch.randint(low=0, high=n, size=(train_size + test_size,), generator=g)
    features, labels = sample_data_from_generator_classes(g, y, rules)

    return features, labels


def sample_data_from_indices(samples, rules, n, m, s, L, return_tree_structure=False):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).

    Returns:
        A tuple containing the inputs and outputs of the model.
    """
    max_data = n * m ** ((s**L-1)//(s-1))
    data_per_hl = max_data // n 	# div by num_classes to get number of data per class

    high_level = samples.div(data_per_hl, rounding_mode='floor')	# div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl					# compute remainder (run in range(data_per_hl))

    labels = high_level	# labels are the classes (features of highest level)
    features = labels		# init input features as labels (rep. size 1)
    size = 1

    if return_tree_structure:
        features_dict = (
            {}
        )  # Initialize the dictionary to store the hidden variables
        features_dict[0] = copy.deepcopy(features)
        for l in range(L):

            choices = m**(size)
            data_per_hl = data_per_hl // choices	# div by num_choices to get number of data per high-level feature

            high_level = low_level.div( data_per_hl, rounding_mode='floor')	# div by data_per_hl to get high-level feature index (1 index in range(m**size))
            high_level = dec2base(high_level, m, length=size).squeeze()	# convert to base m (size indices in range(m), squeeze needed if index already in base m)

            features = rules[l][features, high_level]	        		# apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
            features = features.flatten(start_dim=1)				# flatten to tensor of size (batch_size, size*s)
            features_dict[l+1] = copy.deepcopy(features)
            size *= s								# rep. size increases by s at each level

            low_level = low_level % data_per_hl				# compute remainder (run in range(data_per_hl))

        return features_dict, labels

    else:
        for l in range(L):

            choices = m**(size)
            data_per_hl = data_per_hl // choices	# div by num_choices to get number of data per high-level feature

            high_level = low_level.div( data_per_hl, rounding_mode='floor')	# div by data_per_hl to get high-level feature index (1 index in range(m**size))
            high_level = dec2base(high_level, m, length=size).squeeze()	# convert to base m (size indices in range(m), squeeze needed if index already in base m)

            features = rules[l][features, high_level]	        		# apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
            features = features.flatten(start_dim=1)				# flatten to tensor of size (batch_size, size*s)
            size *= s								# rep. size increases by s at each level

            low_level = low_level % data_per_hl				# compute remainder (run in range(data_per_hl))

        return features, labels


def sample_without_replacement(max_data, train_size, test_size, seed_sample, rules):

    L = len(rules)  # Number of levels in the hierarchy
    n = rules[0].shape[0]  # Number of classes
    m = rules[0].shape[1]  # Number of synonymic lower-level representations
    s = rules[0].shape[2]  # Size of lower-level representations

    if train_size == -1:
        samples = torch.arange(max_data)
    else:
        test_size = min(test_size, max_data - train_size)

        random.seed(seed_sample)
        samples = torch.tensor(random.sample(range(max_data), train_size + test_size))

    features, labels = sample_data_from_indices(samples, rules, n, m, s, L)

    return features, labels

def format_rules(rules): # assumes that n_classes = 1
    rules_string = "S -> "
    last_prob = 1 - (eval(f'{1/len(rules[0][0]):.2f}')*(len(rules[0][0])-1))
    for rule in rules[0][0][:-1]:
        rules_string += f"NT1_{rule[0]} NT1_{rule[1]} [{1/len(rules[0][0]):.2f}] | "
    rules_string += f"NT1_{rules[0][0][-1][0]} NT1_{rules[0][0][-1][1]} [{last_prob:.2f}]\n"

    for L in range(1, len(rules)-1):
        level_rules = rules[L]
        for nt in range(len(level_rules)):
            rules_string += f"NT{L}_{nt} -> "
            for rule in level_rules[nt][:-1]:
                rules_string += f"NT{L+1}_{rule[0]} NT{L+1}_{rule[1]} [{1/len(level_rules[nt]):.2f}] | "
            rules_string += f"NT{L+1}_{level_rules[nt][-1][0]} NT{L+1}_{level_rules[nt][-1][1]} [{last_prob:.2f}]\n"

    L = len(rules)-1
    level_rules = rules[L]
    for nt in range(len(level_rules)):
        rules_string += f"NT{L}_{nt} -> "
        for rule in level_rules[nt][:-1]:
            rules_string += f"'T{rule[0]}' 'T{rule[1]}' [{1/len(level_rules[nt]):.2f}] | "
        rules_string += f"'T{level_rules[nt][-1][0]}' 'T{level_rules[nt][-1][1]}' [{last_prob:.2f}]\n"
    
    return rules_string

class ProbabilisticGenerator(nltk.grammar.PCFG):
    def generate(self, n: int = 1) -> Iterator[str]:
        """Probabilistically, recursively reduce the start symbol `n` times,
        yielding a valid sentence each time.

        Args:
            n: The number of sentences to generate.

        Yields:
            The next generated sentence.
        """
        for _ in range(n):
            x = self._generate_derivation(self.start())
            yield x

    def _generate_derivation(self, nonterminal: nltk.grammar.Nonterminal) -> str:
        """Probabilistically, recursively reduce `nonterminal` to generate a
        derivation of `nonterminal`.

        Args:
            nonterminal: The non-terminal nonterminal to reduce.

        Returns:
            The derived sentence.
        """
        sentence: List[str] = []
        symbol: Symbol
        derivation: str

        for symbol in self._reduce_once(nonterminal):
            if isinstance(symbol, str):
                derivation = symbol
            else:
                derivation = self._generate_derivation(symbol)

            if derivation != "":
                sentence.append(derivation)

        return " ".join(sentence)

    def _reduce_once(self, nonterminal: nltk.grammar.Nonterminal) -> Tuple[Symbol]:
        """Probabilistically choose a production to reduce `nonterminal`, then
        return the right-hand side.

        Args:
            nonterminal: The non-terminal symbol to derive.

        Returns:
            The right-hand side of the chosen production.
        """
        return self._choose_production_reducing(nonterminal).rhs()

    def _choose_production_reducing(
        self, nonterminal: nltk.grammar.Nonterminal
    ) -> nltk.grammar.ProbabilisticProduction:
        """Probabilistically choose a production that reduces `nonterminal`.

        Args:
            nonterminal: The non-terminal symbol for which to choose a production.

        Returns:
            The chosen production.
        """
        productions: List[nltk.grammar.ProbabilisticProduction] = self._lhs_index[nonterminal]
        probabilities: List[float] = [production.prob() for production in productions]
        return random.choices(productions, weights=probabilities)[0]

class RandomHierarchyModel(Dataset):
    """
    Implement the Random Hierarchy Model (RHM) as a PyTorch dataset.
    """

    def __init__(
            self,
            num_features=8,
            num_classes=2,
            num_synonyms=2,
            tuple_size=2,	# size of the low-level representations
            num_layers=2,
            seed_rules=0,
            seed_sample=1,
            train_size=-1,
            test_size=0,
            input_format='onehot',
            whitening=0,
            transform=None,
            replacement=False,
    ):

        self.num_features = num_features
        self.num_synonyms = num_synonyms 
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.tuple_size = tuple_size

        self.rules = sample_rules( num_features, num_classes, num_synonyms, tuple_size, num_layers, seed=seed_rules)
        self.rules_string = format_rules(self.rules)
        self.PCFG = ProbabilisticGenerator.fromstring(self.rules_string)
        self.parser = nltk.ViterbiParser(self.PCFG)
 
        max_data = num_classes * num_synonyms ** ((tuple_size ** num_layers - 1) // (tuple_size - 1))
        assert train_size >= -1, "train_size must be greater than or equal to -1"

        if max_data > 1e19 and not replacement:
            print(
                "Max dataset size cannot be represented with int64! Using sampling with replacement."
            )
            warnings.warn(
                "Max dataset size cannot be represented with int64! Using sampling with replacement.",
                RuntimeWarning,
            )
            replacement = True

        if not replacement:
            self.features, self.labels = sample_without_replacement(
                max_data, train_size, test_size, seed_sample, self.rules
            )
        else:
            self.features, self.labels = sample_with_replacement(
                train_size, test_size, seed_sample, self.rules
            )

        if 'onehot' not in input_format:
            assert not whitening, "Whitening only implemented for one-hot encoding"

	# TODO: implement one-hot encoding of s-tuples
        if 'onehot' in input_format:

            self.features = F.one_hot(
                self.features.long(),
                num_classes=num_features if 'tuples' not in input_format else num_features ** tuple_size
            ).float()
            
            if whitening:

                inv_sqrt_norm = (1.-1./num_features) ** -.5
                self.features = (self.features - 1./num_features) * inv_sqrt_norm

            self.features = self.features.permute(0, 2, 1)

        elif 'long' in input_format:
            self.features = self.features.long()

        else:
            raise ValueError

        self.transform = transform

        self.bos = self.features.max() + 1
        self.eos = self.features.max() + 2
        self.max_sample_length = self.features.size(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
        	idx: sample index

        Returns:
            Feature-label pairs at index            
        """
        x, y = self.features[idx], self.labels[idx]

        if self.transform:
            x, _ = self.transform(x, y)

        return (torch.cat([torch.tensor([self.bos]), x, torch.tensor([self.eos])], dim=0),
                torch.tensor(self.tuple_size ** self.num_layers, dtype=torch.float))

    def get_rules(self):
        return self.rules

    def check_grammaticality(self, sentence: str) -> bool:
        """Check if a sentence is in the grammar.

        Args:
            sentence: The sentence to check.

        Returns:
            Whether the sentence is in the grammar.
        """

        # Remove instruction decorator and pad tokens
        if f'T{self.bos}' in sentence:
            sentence = sentence.split(f'T{self.bos}')
            sentence = sentence[1] if len(sentence) > 1 else sentence[0]
 
        # Tokenize the sentence
        tokens = sentence.split(' ')
        while '' in tokens:
            tokens.remove('')

        # Run parser
        try:
            parser_output = self.parser.parse(tokens).__next__()
            logprobs, height = parser_output.logprob(), parser_output.height()
            return (True, logprobs, height, None), len(tokens)
        except:
            failure = ' '.join(tokens)
            return (False, None, None, failure), len(tokens)

    def save_grammar(self, path_to_results: str):
        """
        Save the grammar underlying the dataset
        """
        base_dir = os.path.join(path_to_results, 'grammar')
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, 'PCFG.pkl'), 'wb') as f:
            pkl.dump(self.PCFG, f)
