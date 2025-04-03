import networkx as nx
from typing import List, Set, Tuple
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
        sub_order = order[i+1:]
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
