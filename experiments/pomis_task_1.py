import numpy as np
from itertools import product, combinations


class Task1SCM:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.variables = ['Z1', 'Z2', 'X1', 'X2', 'Y']
        self.parent_dict = {
            'Z1': [],
            'Z2': [],
            'X1': ['Z1'],
            'X2': ['Z2'],
            'Y': ['X1', 'X2']
        }

    def sample_node(self, name, sample):
        if name == 'Z1' or name == 'Z2':
            return self.rng.integers(0, 2)
        if name == 'X1':
            return sample['Z1']
        if name == 'X2':
            return sample['Z2']
        if name == 'Y':
            return sample['X1'] ^ sample['X2']  # XOR = optimal when both 1
        raise ValueError(f"Unknown node {name}")

    def sample_observation(self):
        sample = {}
        for v in self.variables:
            sample[v] = self.sample_node(v, sample)
        return sample, sample['Y']

    def intervene_and_sample(self, action_dict):
        sample = {}
        for v in self.variables:
            if v in action_dict:
                sample[v] = action_dict[v]
            else:
                sample[v] = self.sample_node(v, sample)
        return sample, sample['Y']

    def get_all_binary_assignments(self, vars):
        return [dict(zip(vars, values)) for values in product([0, 1], repeat=len(vars))]

    def get_action_space(self, strategy='brute-force'):
        V = [v for v in self.variables if v != 'Y']
        arms = []
        if strategy == 'all-at-once':
            return self.get_all_binary_assignments(V)
        elif strategy == 'brute-force':
            for r in range(1, len(V) + 1):
                for subset in map(list, combinations(V, r)):
                    arms.extend(self.get_all_binary_assignments(subset))
            return arms
        elif strategy == 'mis':
            # Placeholder — will refine later
            return self.get_all_binary_assignments(['X1', 'X2'])
        elif strategy == 'pomis':
            # Placeholder — will refine when Alg. 1 is implemented
            return self.get_all_binary_assignments(['X1', 'X2'])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def true_optimal_arm(self):
        return {'X1': 1, 'X2': 1}  # This gives Y = 0 (XOR), assuming optimal reward is 0

    def true_reward(self, action):
        _, y = self.intervene_and_sample(action)
        return y
