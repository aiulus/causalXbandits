import sys
from abc import ABC
from pathlib import Path

# Add submodule repo paths
sys.path.append(str(Path(__file__).resolve().parent.parent / "mab"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "causal-models"))

from causal_models.scm.base import SCM
from mab.bandits.base import Bandit


class SCMGovernedBandit(Bandit, ABC):
    def __init__(self, scm: SCM, reward_node: str, action_nodes: list, seed=None):
        self.scm = scm
        self.reward_node = reward_node
        self.action_nodes = action_nodes
        super().__init__(n_arms=len(action_nodes), seed=seed)

    def pull(self, arm: int) -> float:
        # Intervene on action_nodes[arm], sample SCM, extract reward_node
        do_intervention = {self.action_nodes[arm]: 1}
        self.scm.intervene(do_intervention)
        sample = self.scm.sample(1, mode='interventional')
        return float(sample[self.reward_node][0])
