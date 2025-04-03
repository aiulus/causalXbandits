import numpy as np
from itertools import product
from mab.algorithms.ucb import UCB


class RAPS_UCB:
    def __init__(self, scm, T: int, delta: float = 0.01, epsilon: float = 0.05, B: int = None):
        """
        Args:
            scm: Structural Causal Model with .intervene_and_sample(action_dict)
            T: Total time horizon
            delta: Confidence level for RAPS
            epsilon: Threshold for detecting dependencies
            B: Number of samples per RAPS check (None = auto-computed from theory)
        """
        self.scm = scm
        self.T = T
        self.delta = delta
        self.epsilon = epsilon
        self.B = B
        self.parent_set = None
        self.ucb = None
        self.arm_list = []
        self.arm_index = {}  # map from action dict to index

        self.regret = []
        self.best_action = None

    def run(self):
        # --- Phase 1: Parent Discovery ---
        self.parent_set = self.discover_all_parents()
        print(f"Discovered parent set: {self.parent_set}")

        # --- Phase 2: Setup UCB over 2^|P| arms ---
        self.arm_list = self.generate_parent_assignments(self.parent_set)
        self.arm_index = {frozenset(a.items()): i for i, a in enumerate(self.arm_list)}

        self.ucb = UCB(n_arms=len(self.arm_list), horizon=self.T, delta=self.delta)

        # --- Phase 3: UCB Exploration ---
        for t in range(self.T):
            arm_idx = self.ucb.select_arm()
            action = self.arm_list[arm_idx]
            _, reward = self.scm.intervene_and_sample(action)
            self.ucb.update(arm_idx, reward)
            self.regret.append(reward)  # Optionally compute regret later

        # --- Best action (empirical) ---
        self.best_action = self.arm_list[np.argmax(self.ucb.means)]
        print(f"Best empirical action: {self.best_action}")

    def generate_parent_assignments(self, parent_set):
        return [dict(zip(parent_set, values)) for values in product([0, 1], repeat=len(parent_set))]

    def discover_all_parents(self):
        """
        Plug in your implementation of Algorithm 2 here.
        Should return a list of variable names like ['X1', 'X2'].
        """
        # Replace this call with your actual implementation
        from algorithms.raps import RAPS
        return RAPS.discover_all_parents(self.scm, B=self.B, delta=self.delta, epsilon=self.epsilon)

    def get_average_reward(self):
        return np.mean(self.regret)
