from typing import Iterator, List, Tuple, Union
import random
import numpy as np
import nltk  # type: ignore
from nltk.grammar import ProbabilisticProduction  # type: ignore
from nltk.grammar import Nonterminal  # type: ignore
from .utils import define_prior

Symbol = Union[str, Nonterminal]


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

    def _generate_derivation(self, nonterminal: Nonterminal) -> str:
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

    def _reduce_once(self, nonterminal: Nonterminal) -> Tuple[Symbol]:
        """Probabilistically choose a production to reduce `nonterminal`, then
        return the right-hand side.

        Args:
            nonterminal: The non-terminal symbol to derive.

        Returns:
            The right-hand side of the chosen production.
        """
        return self._choose_production_reducing(nonterminal).rhs()

    def _choose_production_reducing(
        self, nonterminal: Nonterminal
    ) -> ProbabilisticProduction:
        """Probabilistically choose a production that reduces `nonterminal`.

        Args:
            nonterminal: The non-terminal symbol for which to choose a production.

        Returns:
            The chosen production.
        """
        productions: List[ProbabilisticProduction] = self._lhs_index[nonterminal]
        probabilities: List[float] = [production.prob() for production in productions]
        return random.choices(productions, weights=probabilities)[0]



class PCFG:

    def __init__(
            self,
            language : str = 'english', # in ['english', 'expr', 'dyck']
            config : dict = {'n_nouns': 10,
                             'n_verbs': 10,
                             'n_adjectives': 10,
                             'n_pronouns': 10,
                             'n_adverbs': 10,
                             'n_conjunctions': 2,
                             'p_conjunctions': 0.15,
                             'n_prepositions': 0,
                             'relative_clauses': False,
                             'transitive_verbs': False}, # config depends on the language; see below
            alpha: float = 1e5,
            prior_type: str = 'dirichlet',
            tasks: dict = None,
            seed: int = 42,
            ):
        """Define the PCFG object.

        Args:
            language: The language of the PCFG. One of ['english', 'expr', 'dyck1', 'dyck2'].
            config: The configuration of the PCFG. The keys depend on the language.
            * For 'english':
                n_nouns: The number of nouns in the vocabulary.
                n_verbs: The number of verbs in the vocabulary.
                n_adjectives: The number of adjectives in the vocabulary.
                n_pronouns: The number of pronouns in the vocabulary.
                n_adverbs: The number of adverbs in the vocabulary.
                n_conjunctions: The number of conjunctions in the vocabulary.
                p_conjunctions: The probability of generating a conjunction.
                n_prepositions: The number of prepositions in the vocabulary.
                relative_clauses: Whether to generate relative clauses (as both adjectives and adverbs).
                transitivity: Whether to distinguish transitive and intransitive verbs.
            * For 'expr':
                n_digits: The number of digits in the vocabulary.
                n_ops: The number of operations in the vocabulary.
                postfix: Whether the grammar is postfix or prefix.
            * For 'dyck':
                n_brackets: The number of types brackets in the vocabulary.
                p_nest: The probability of nesting sequences. Should be ≤ 0.5.
            alpha: The concentration parameter for the Dirichlet distribution.
            prior_type: The type of prior distribution.
            tasks: The tasks to perform.
            seed: The random seed.

        Returns:
            PCFG: A PCFG object.
        """

        # Set the random seed
        random.seed(seed)
        np.random.seed(seed)

        self.language = language
        self.alpha = alpha
        self.prior_type = prior_type
        
        # Grammar
        self.production_rules = None
        self.lexical_symbolic_rules = None

        # Concept classes object
        if language == 'english':
            self.n_nouns = config['n_nouns']
            self.n_verbs = config['n_verbs']
            self.n_adjectives = config['n_adjectives']
            self.n_pronouns = config['n_pronouns']
            self.n_adverbs = config['n_adverbs']
            self.n_conjunctions = config['n_conjunctions']
            self.p_conjunctions = config['p_conjunctions'] if 'p_conjunctions' in config else 0.15
            self.n_prepositions = config['n_prepositions'] if 'n_prepositions' in config else 0
            self.relative_clauses = config['relative_clauses'] if 'relative_clauses' in config else False
            self.transitivity = config['transitivity'] if 'transitivity' in config else False
            self.grammar = self.create_grammar_english(
                n_nouns=self.n_nouns,
                n_verbs=self.n_verbs,
                n_adjectives=self.n_adjectives,
                n_pronouns=self.n_pronouns,
                n_adverbs=self.n_adverbs,
                n_conjunctions=self.n_conjunctions,
                p_conjunctions=self.p_conjunctions,
                n_prepositions=self.n_prepositions,
                relative_clauses=self.relative_clauses,
                transitivity=self.transitivity,
                )

        elif language == 'expr':
            self.n_digits = config['n_digits']
            self.n_ops = config['n_ops']
            self.postfix = config['postfix']
            self.grammar = self.create_grammar_expr(
                n_digits=self.n_digits,
                n_ops=self.n_ops,
                postfix=self.postfix,
                )

        elif language == 'dyck':
            self.n_brackets = config['n_brackets']
            self.p_nest = config['p_nest'] if 'p_nest' in config else 0.50
            self.grammar = self.create_grammar_dyck(
                n_brackets=self.n_brackets,
                p_nest=self.p_nest
            )

        else:
            raise ValueError(f"Language {language} not supported. Options are ['english', 'expr', 'dyck1', 'dyck2'].")

        # Tasks
        self.tasks = tasks

        # Set the vocabulary
        self.vocab, self.id_to_token_map, self.vocab_size = self.gather_vocabulary()

        # Parser
        self.parser = nltk.ViterbiParser(self.grammar)


    def create_grammar_english(
            self, 
            n_nouns: int,
            n_verbs: int,
            n_adjectives: int,
            n_pronouns: int,
            n_adverbs: int,
            n_conjunctions: int,
            p_conjunctions: float,
            n_prepositions: int,
            relative_clauses: bool,
            transitivity: bool,
            ):
        """Define the PCFG grammar.

        Args:
            n_nouns: The number of nouns in the vocabulary.
            n_verbs: The number of verbs in the vocabulary.
            n_adjectives: The number of adjectives in the vocabulary.
            n_pronouns: The number of pronouns in the vocabulary.
            n_adverbs: The number of adverbs in the vocabulary.
            n_conjunctions: The number of conjunctions in the vocabulary.
            p_conjunctions: The probability of generating a conjunction.
            n_prepositions: The number of prepositions in the vocabulary.
            relative_clauses: Whether to generate relative clauses (as both adjectives and adverbs).
            transitivity: Whether to distinguish transitive and intransitive verbs.

        Returns:
            The PCFG grammar.
        """

        # Define production rules
        self.production_rules = """
                S -> NP VP [1.0]
                """

        # Define expansions of required non-terminals
        np_expansions = {'Pro': 0.20,   'N': 0.20,  'NP Conj NP': p_conjunctions, 'Adj N': 0, 'NP AdjRel': 0, 'NP PP': 0}
        vp_expansions = {'VP Conj VP': p_conjunctions, 'VP Adv': 0, 'VP AdvRel': 0, 'VP PP': 0}
        vp_expansions.update({'TV NP': 0.20, 'IV': 0.20} if transitivity else {'V NP': 0.20,  'V': 0.20})
        expansions = {'NP': np_expansions, 'VP': vp_expansions}

        if relative_clauses:
            adjrel_expansions = {'RP TV NP': 0.33, 'RP IV': 0.33, 'RP NP TV': 0.34} if transitivity else \
                                {'RP V NP': 0.33,  'RP V': 0.33,  'RP NP V': 0.34}
            advrel_expansions = {'RA S': 1}
            expansions.update({'AdjRel': adjrel_expansions, 'AdvRel': advrel_expansions})
        if n_prepositions > 0:
            pp_expansions = {'P NP': 1}
            expansions.update({'PP': pp_expansions})
        
        n = 1 + sum([relative_clauses, n_prepositions > 0]) # 1 for Adj N or VP Adv
        p = eval(f'{(0.6 - p_conjunctions)/n:0.2f}') # 0.4 for Pro, N or TV, IV
        # Assign probabilities to NP expansions
        np_expansions['Adj N'] = p
        if relative_clauses: np_expansions['NP AdjRel'] = p
        if n_prepositions > 0: np_expansions['NP PP'] = p
        if sum(np_expansions.values()) < 1: np_expansions['N'] += 1 - sum(np_expansions.values())

        # Assign probabilities to VP expansions
        vp_expansions['VP Adv'] = p
        if relative_clauses: vp_expansions['VP AdvRel'] = p
        if n_prepositions > 0: vp_expansions['VP PP'] = p
        remaining_p = 1 - sum(vp_expansions.values())
        if remaining_p > 0:
            if transitivity: vp_expansions['IV'] += remaining_p
            else: vp_expansions['V'] += remaining_p

        # Format the expansions
        for nonterminal, exps in expansions.items():
            rhs_symbol = ""
            for rhs, prob in exps.items():
                rhs_symbol += f"{rhs} [{prob}] | "
            rhs_symbol = rhs_symbol[:-3]
            self.production_rules += f"{nonterminal} -> {rhs_symbol} \n"

        self.lexical_symbolic_rules = ""

        ## Define lexical rules)
        symbol_types = ['N'] + (['TV', 'IV'] if transitivity else ['V']) + ['Adj', 'Pro', 'Adv', 'Conj']
        n_symbol_to_tokens = [n_nouns] + ([n_verbs // 2,  (n_verbs - n_verbs // 2)] if self.transitivity else [n_verbs]) + [n_adjectives, n_pronouns, n_adverbs, n_conjunctions]
        token_prefix = ['noun'] + (['tverb', 'iverb'] if transitivity else ['verb']) + ['adj', 'pro', 'adv', 'conj']
        if n_prepositions > 0:
            symbol_types += ['P']
            n_symbol_to_tokens += [n_prepositions]
            token_prefix += ['prep']
        if relative_clauses:
            symbol_types += ['RP', 'RA']
            n_symbol_to_tokens += [2, 2]
            token_prefix += ['relp', 'rela']

        for symbol_type, n_symbol_to_token, prefix in zip(symbol_types, n_symbol_to_tokens, token_prefix):
            prior_over_symbol = define_prior(n_symbol_to_token, alpha=self.alpha, prior_type=self.prior_type)
            rhs_symbol = ""
            for i in range(n_symbol_to_token):
                rhs_symbol += f"'{prefix}{i}' [{prior_over_symbol[i]}] | "
            rhs_symbol = rhs_symbol[:-3]
            self.lexical_symbolic_rules += f"{symbol_type} -> {rhs_symbol} \n"
        
        # Create the grammar
        return ProbabilisticGenerator.fromstring(self.production_rules + self.lexical_symbolic_rules)

    def create_grammar_expr(
            self, 
            n_digits: int,
            n_ops: int,
            postfix: bool,
            ):
        """Define the PCFG grammar.

        Args:
            n_digits: The number of digits in the vocabulary.
            n_ops: The number of operations in the vocabulary.
            postfix: Whether the grammar is postfix or prefix.

        Returns:
            The PCFG grammar.
        """

        # Define production rules
        self.production_rules = """
                S -> Expr [1.0]
                Expr -> OpExpr [0.40] | Digit [0.60]"""
        if postfix:
            self.production_rules += """
                    OpExpr -> Expr UnOp [0.33] | Expr Expr BinOp [0.33] | Expr Expr Expr TernOp [0.34]
                    """
        else:
            self.production_rules += """
                    OpExpr -> UnOp Expr [0.33] | BinOp Expr Expr [0.33] | TernOp Expr Expr Expr [0.34]
                    """
        
        self.lexical_symbolic_rules = ""

        ## Define lexical rules
        symbol_types = ['Digit', 'UnOp', 'BinOp', 'TernOp']
        n_symbol_to_tokens = [n_digits, n_ops, n_ops, n_ops]
        token_prefix = ['dig', 'un', 'bin', 'tern']

        for symbol_type, n_symbol_to_token, prefix in zip(symbol_types, n_symbol_to_tokens, token_prefix):
            prior_over_symbol = define_prior(n_symbol_to_token, alpha=self.alpha, prior_type=self.prior_type)
            rhs_symbol = ""
            for i in range(n_symbol_to_token):
                rhs_symbol += f"'{prefix}{i}' [{prior_over_symbol[i]}] | "
            rhs_symbol = rhs_symbol[:-3]
            self.lexical_symbolic_rules += f"{symbol_type} -> {rhs_symbol} \n"

        # Create the grammar
        return ProbabilisticGenerator.fromstring(self.production_rules + self.lexical_symbolic_rules)

    def create_grammar_dyck(
            self, 
            n_brackets: int,
            p_nest: float
            ):
        """Define the PCFG grammar.

        Args:
            n_brackets: The number of types brackets in the vocabulary.
            p_nest: The probability of nesting sequences. Should be ≤ 0.5.

        Returns:
            The PCFG grammar.
        """

        # Define production rules
        p = 0.30
        # Probability of generating a new bracket.
        # This is the highest probability that doesn't lead to infinite recursion.
        self.production_rules = f"""
                S -> S S [{p}]"""
        remaining_p = 1 - p

        for i in range(n_brackets-1):
            self.production_rules += f" | Brack{i} [{remaining_p/n_brackets:0.2f}]"
            p += eval(f'{remaining_p/n_brackets:0.2f}')
        self.production_rules += f" | Brack{n_brackets-1} [{1-p}]\n"

        for i in range(n_brackets):
            self.production_rules += f"Brack{i} -> 'o{i}' S 'c{i}' [{p_nest:.2f}] | 'o{i}' 'c{i}' [{1-p_nest:.2f}]\n"
        
        self.lexical_symbolic_rules = ""

        # Create the grammar
        return ProbabilisticGenerator.fromstring(self.production_rules + self.lexical_symbolic_rules)


    def gather_vocabulary(self):
        """Gather the vocabulary from the concept classes.

        Returns:
            The vocabulary.
        """

        # Gather concept classes' vocabulary
        vocab = {}
        vocab_size = 0
        if self.language == 'english':
            n_symbol_to_tokens = [self.n_nouns] + ([self.n_verbs // 2,  (self.n_verbs - self.n_verbs // 2)] if self.transitivity else [self.n_verbs]) + [self.n_adjectives, self.n_pronouns, self.n_adverbs, self.n_conjunctions]
            token_prefix = ['noun'] + (['tverb', 'iverb'] if self.transitivity else ['verb']) + ['adj', 'pro', 'adv', 'conj']
            if self.n_prepositions > 0:
                n_symbol_to_tokens += [self.n_prepositions]
                token_prefix += ['prep']
            if self.relative_clauses:
                n_symbol_to_tokens += [2, 2]
                token_prefix += ['relp', 'rela']
            for prefix, n_symbol_to_token in zip(token_prefix, n_symbol_to_tokens):
                for i in range(n_symbol_to_token):
                    vocab[f'{prefix}{i}'] = vocab_size
                    vocab_size += 1
        elif self.language == 'expr':
            n_symbol_to_tokens = [self.n_digits, self.n_ops, self.n_ops, self.n_ops]
            token_prefix = ['dig', 'un', 'bin', 'tern']
            for prefix, n_symbol_to_token in zip(token_prefix, n_symbol_to_tokens):
                for i in range(n_symbol_to_token):
                    vocab[f'{prefix}{i}'] = vocab_size
                    vocab_size += 1
        elif self.language == 'dyck':
            n_symbol_to_tokens = [self.n_brackets, self.n_brackets]
            token_prefix = ['o', 'c']
            for prefix, n_symbol_to_token in zip(token_prefix, n_symbol_to_tokens):
                for i in range(n_symbol_to_token):
                    vocab[f'{prefix}{i}'] = vocab_size
                    vocab_size += 1
        vocab_size = len(vocab)

        # Add special tokens to be used for defining sequences in dataloader
        for special_token in ['<pad>', 'Task:', '<null>', 'Ops:', 'Out:', '\n', '<eos>', '<sep>']:
            vocab[special_token] = vocab_size
            vocab_size += 1

        # Add task tokens
        for task_token in self.tasks:
            vocab[task_token] = vocab_size
            vocab_size += 1

        # Create an inverse vocabulary
        id_to_token_map = {v: k for k, v in vocab.items()}

        return vocab, id_to_token_map, vocab_size


    def tokenize_sentence(self, sentence: str) -> List[int]:
        """Tokenize a sentence.

        Args:
            sentence: The sentence to tokenize.

        Returns:
            The tokenized sentence.
        """

        # Tokenize the sentence
        tokens = sentence.split(' ')

        # Convert the tokens to indices
        token_indices = []
        for token in tokens:
            if token == '' or token == ' ':
                continue
            else:
                token_indices.append(self.vocab[token])

        return token_indices


    def detokenize_sentence(self, token_indices) -> str:
        """Detokenize a sentence.

        Args:
            token_indices: The token indices to detokenize.

        Returns:
            The detokenized sentence.
        """

        # Convert the indices to tokens
        tokens = [self.id_to_token_map[token] for token in np.array(token_indices)]

        # Detokenize the tokens
        sentence = " ".join(tokens)

        return sentence


    def sentence_generator(
            self, 
            num_of_samples: int,
            ) -> Iterator[str]:
        """
        1. Generate a sentence from the grammar
        2. Fill the sentence with values from the concept classes
        """

        # An iterator that dynamically generates symbolic sentences from the underlying PCFG
        symbolic_sentences = self.grammar.generate(num_of_samples)

        # Fill the sentences with values from the concept classes
        for s in symbolic_sentences:
            yield s



    def check_grammaticality(self, sentence: str) -> bool:
        """Check if a sentence is in the grammar.

        Args:
            sentence: The sentence to check.

        Returns:
            Whether the sentence is in the grammar.
        """

        # Remove instruction decorator and pad tokens
        if 'Out:' in sentence:
            sentence = sentence.split('Out: ')
            sentence = sentence[1] if len(sentence) > 1 else sentence[0]
        if '<pad>' in sentence:
            sentence = sentence.split(' <pad>')
            sentence = sentence[0] if len(sentence) > 1 else sentence[0]

        # Tokenize the sentence
        tokens = sentence.split(' ')
        if '' in tokens:
            tokens.remove('')

        # Run parser
        try:
            parser_output = self.parser.parse(tokens).__next__()
            logprobs, height = parser_output.logprob(), parser_output.height()
            return (True, logprobs, height, None), len(tokens)
        except:
            failure = ' '.join(tokens)
            return (False, None, None, failure), len(tokens)
