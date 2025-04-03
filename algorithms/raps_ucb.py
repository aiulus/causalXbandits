# @article{konobeev2023causal,
#   title={Causal bandits without graph learning},
#   author={Konobeev, Mikhail and Etesami, Jalal and Kiyavash, Negar},
#   journal={arXiv preprint arXiv:2301.11401},
#   year={2023}
# }

import math
from itertools import product

class RAPS_UCB:
    def __init__(self, scm, raps, T=1000, delta=0.01, K=2):
        self.scm = scm
        self.raps = raps  # Instance of RAPS
        self.T = T
        self.delta = delta
        self.K = K
        self.parent_nodes = []
        self.arm_values = []
        self.arm_counts = {}
        self.arm_rewards = {}

    def initialize_ucb(self):
        self.arm_counts = {a: 0 for a in self.arm_values}
        self.arm_rewards = {a: 0.0 for a in self.arm_values}

    def select_ucb_arm(self, t):
        """Select arm using UCB1 rule."""
        log_term = math.log(t + 1)
        ucb_scores = {}
        for a in self.arm_values:
            count = self.arm_counts[a]
            mean = self.arm_rewards[a] / count if count > 0 else 0
            bonus = math.sqrt(2 * log_term / count) if count > 0 else float('inf')
            ucb_scores[a] = mean + bonus
        return max(ucb_scores, key=ucb_scores.get)

    def run(self):
        print(">>> Discovering parent nodes using RAPS...")
        self.parent_nodes = self.raps.discover_all_parents()
        print(f">>> Discovered parent nodes: {self.parent_nodes}")

        # Build the reduced action space over do(P=x)
        self.arm_values = list(product(range(self.K), repeat=len(self.parent_nodes)))
        self.initialize_ucb()

        print(f">>> Starting UCB loop with {len(self.arm_values)} arms...")
        for t in range(1, self.T + 1):
            arm = self.select_ucb_arm(t)
            intervention = dict(zip(self.parent_nodes, arm))
            _, reward = self.scm.intervene_and_sample(intervention)

            self.arm_counts[arm] += 1
            self.arm_rewards[arm] += reward

        # Estimate best arm
        best_arm = max(self.arm_values, key=lambda a: self.arm_rewards[a] / self.arm_counts[a])
        return dict(zip(self.parent_nodes, best_arm))
