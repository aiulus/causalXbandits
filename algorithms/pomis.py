# @article{lee2018structural,
#   title={Structural causal bandits: Where to intervene?},
#   author={Lee, Sanghack and Bareinboim, Elias},
#   journal={Advances in neural information processing systems},
#   volume={31},
#   year={2018}
# }

import networkx as nx
from typing import List, Set, Tuple
from collections import defaultdict
from causal_models.graphs.utils import (
    get_ancestors,
    get_descendants,
    c_component,
    reversed_topological,
    induce_subgraph,
)


def find_muct_and_ib(G: nx.DiGraph, Y: str) -> Tuple[Set[str], Set[str]]:
    """
    Compute the minimal unobserved confounder territory (MUCT) and interventional border (IB)
    relative to the reward variable Y.
    """
    H = induce_subgraph(G, get_ancestors(G, Y))
    T = {Y}

    # Expand via bidirected c-components and descendants
    changed = True
    while changed:
        changed = False
        new_T = set(T)
        for comp in c_component(H):
            if T & comp:
                new_T |= comp
        new_T |= get_descendants(H, new_T)
        if new_T != T:
            changed = True
            T = new_T

    IB = set()
    for node in T:
        for parent in G.predecessors(node):
            if parent not in T:
                IB.add(parent)

    return T, IB


def find_pomis(G: nx.DiGraph, Y: str) -> List[Set[str]]:
    """
    Implements the recursive algorithm from the POMIS paper (Alg. 1 and 2)
    to find all Possibly-Optimal Minimal Intervention Sets.
    """
    T, X = find_muct_and_ib(G, Y)
    H = induce_subgraph(G, T.union(X))
    topo_order = reversed_topological(H, T - {Y})

    return [X] + _subpomis(H, Y, topo_order, blocked=T - {Y})


def _subpomis(G: nx.DiGraph, Y: str, order: List[str], blocked: Set[str]) -> List[Set[str]]:
    P = []
    for i, v in enumerate(order):
        sub_order = order[i + 1:]
        O_prime = blocked.union(order[:i])
        G_v = G.copy()
        G_v.remove_edges_from(list(G_v.in_edges(v)))

        T, X = find_muct_and_ib(G_v, Y)

        if not X & O_prime:
            P.append(X)
            if sub_order:
                H = induce_subgraph(G_v, T.union(X))
                P += _subpomis(H, Y, sub_order, O_prime)
    return P


def get_estimand_sets(G, target):
    """
    Returns a mapping from each node to its conditioning set (Zj),
    needed for the POMIS-style estimation of P(target | do(Xi = x)).
    Assumes that bidirected confounding is marked via the 'confounded' attribute.
    """
    def get_c_component(node):
        """Find all nodes in the same confounded component as `node`."""
        component = set()
        for u, v, data in G.edges(data=True):
            if data.get("confounded"):
                if u == node:
                    component.add(v)
                elif v == node:
                    component.add(u)
        component.add(node)
        return component

    def get_parents(node):
        """Return parents of `node`, excluding confounded edges."""
        return set(p for p in G.predecessors(node) if not G.edges[p, node].get("confounded", False))

    estimand_sets = {}

    for node in G.nodes:
        Cj = get_c_component(node)
        parents_union = set()
        for cj_node in Cj:
            parents_union |= get_parents(cj_node)
        Zj = (parents_union | Cj) - {node}
        estimand_sets[node] = Zj

    return estimand_sets
