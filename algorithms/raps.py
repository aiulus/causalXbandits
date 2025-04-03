# @article{konobeev2023causal,
#   title={Causal bandits without graph learning},
#   author={Konobeev, Mikhail and Etesami, Jalal and Kiyavash, Negar},
#   journal={arXiv preprint arXiv:2301.11401},
#   year={2023}
# }

import numpy as np
from typing import Dict, List, Set, Optional
from collections import defaultdict
from itertools import product


class RAPS:
    def __init__(self, scm, K=2, epsilon=0.1, delta=0.01, B=None, seed=None):
        """
        scm: structural causal model object
        K: number of possible values each variable can take (default: binary)
        epsilon, delta: statistical thresholds
        B: number of samples per intervention to estimate mean values
        """
        self.scm = scm
        self.K = K
        self.epsilon = epsilon
        self.delta = delta
        self.B = B if B is not None else int(8 / epsilon ** 2 * np.log(8 * scm.n ** 2 * K ** 2 / delta))
        self.rng = np.random.default_rng(seed)
        self.all_nodes = [v for v in self.scm.variables if v != 'R']

    def estimate_reward_mean(self, intervention: Dict) -> float:
        """Estimate mean reward under intervention `do(intervention)`."""
        rewards = [self.scm.intervene_and_sample(intervention)[1] for _ in range(self.B)]
        return np.mean(rewards)

    def estimate_descendants(self, node: str, parent_set: List[str]) -> Set[str]:
        """
        Estimate descendants of node using differences in distributions under interventions.
        Based on: section 5.1 of the paper.
        """
        empirical_distributions = defaultdict(lambda: defaultdict(int))

        for x_val in range(self.K):
            for z_vals in product(range(self.K), repeat=len(parent_set)):
                intervention = {node: x_val}
                intervention.update(dict(zip(parent_set, z_vals)))

                for _ in range(self.B):
                    sample, _ = self.scm.intervene_and_sample(intervention)
                    for v in self.all_nodes:
                        val = sample[v]
                        empirical_distributions[(v, x_val, z_vals)][val] += 1

        descendants = set()
        for v in self.all_nodes:
            if v == node:
                continue
            for x_val in range(self.K):
                for z_vals in product(range(self.K), repeat=len(parent_set)):
                    counts = empirical_distributions[(v, x_val, z_vals)]
                    total = sum(counts.values())
                    if total == 0:
                        continue
                    probs = {val: count / total for val, count in counts.items()}
                    # naive total variation distance check
                    for val in probs:
                        if abs(probs[val] - 0.5) > self.epsilon / 2:
                            descendants.add(v)
        return descendants

    def intervene_and_check_parent(self, node: str, parent_set: List[str]) -> bool:
        """Checks if node is likely a parent (causal ancestor of reward node)."""
        for x_val in range(self.K):
            for z_vals in product(range(self.K), repeat=len(parent_set)):
                intervention = {node: x_val}
                intervention.update(dict(zip(parent_set, z_vals)))
                r1 = self.estimate_reward_mean(intervention)

                # Flip value of `node`
                x_val_alt = 1 - x_val if self.K == 2 else (x_val + 1) % self.K
                intervention[node] = x_val_alt
                r2 = self.estimate_reward_mean(intervention)

                if abs(r1 - r2) > self.epsilon / 2:
                    return True
        return False

    def rec(self, C: Set[str], parent_set: List[str]) -> Optional[str]:
        if not C:
            return None

        X = self.rng.choice(list(C))
        is_parent = self.intervene_and_check_parent(X, parent_set)
        descendants = self.estimate_descendants(X, parent_set)

        if is_parent:
            rest = descendants - {X}
            P_hat = self.rec(rest, parent_set)
            return X if P_hat is None else P_hat
        else:
            return self.rec(C - descendants, parent_set)

    def find_one_parent(self, candidate_set: Optional[Set[str]] = None, parent_set: Optional[List[str]] = None):
        C = candidate_set if candidate_set is not None else set(self.all_nodes)
        P = parent_set if parent_set is not None else []
        return self.rec(C, P)

    def discover_all_parents(self) -> List[str]:
        P = []
        S = set()
        C = set(self.all_nodes)
        while True:
            P_hat = self.find_one_parent(candidate_set=C, parent_set=P)
            if P_hat is None:
                break
            P.append(P_hat)
            # Update search space
            desc = self.estimate_descendants(P_hat, P)
            S.update(desc)
            C = set(self.all_nodes) - S
        return P
