import networkx as nx
from itertools import chain, combinations
from typing import List, Set, Tuple, Dict
from causal_models.graphs.utils import get_ancestors

def find_muct(G: nx.DiGraph, Y: str) -> Set[str]:
    H = G.subgraph(get_ancestors(G, Y)).copy()
    # Use bidirected edges to grow a MUCT
    muct = {Y}
    changed = True
    while changed:
        changed = False
        descendants = get_descendants(H, muct)
        cc = set()
        for c_set in c_component(H):
            if muct.intersection(c_set):
                cc.update(c_set)
        new = cc.union(descendants)
        if not new.issubset(muct):
            muct.update(new)
            changed = True
    return muct


def get_interventional_border(G: nx.DiGraph, muct: Set[str]) -> Set[str]:
    parents = set()
    for node in muct:
        for p in G.predecessors(node):
            if p not in muct:
                parents.add(p)
    return parents


def induce_subgraph(G: nx.DiGraph, nodes: Set[str]) -> nx.DiGraph:
    return G.subgraph(nodes).copy()


def pomiss(G: nx.DiGraph, Y: str) -> List[Set[str]]:
    T = find_muct(G, Y)
    X = get_interventional_border(G, T)
    H = induce_subgraph(G, T.union(X))
    π = reversed_topological(H, T - {Y})
    found = {frozenset(X)}
    found |= sub_pomiss(H, Y, π, set())
    return [set(s) for s in found]


def sub_pomiss(G: nx.DiGraph, Y: str, π: List[str], O: Set[str]) -> Set[frozenset]:
    results = set()
    for i, πi in enumerate(π):
        G_i = G.copy()
        G_i.remove_edges_from(list(G_i.in_edges(πi)))
        T = find_muct(G_i, Y)
        X = get_interventional_border(G_i, T)
        π_next = π[i+1:]
        O_next = O.union(π[:i])
        if not X.intersection(O_next):
            results.add(frozenset(X))
            if π_next:
                subgraph = induce_subgraph(G_i, T.union(X))
                results |= sub_pomiss(subgraph, Y, π_next, O_next)
    return results


# Example use: KL-UCB bandit with arms from POMIS
def pomis_kl_ucb(
    bandit_env,
    G: nx.DiGraph,
    Y: str,
    horizon: int,
    kl_ucb_fn,
    value_domain: Dict[str, List]
):
    # Get valid intervention sets (POMIS)
    pomis_sets = pomiss(G, Y)

    # Enumerate all do interventions over these sets
    def intervention_space(vars):
        from itertools import product
        doms = [value_domain[v] for v in vars]
        for vals in product(*doms):
            yield dict(zip(vars, vals))

    arms = []
    for pomis_set in pomis_sets:
        arms.extend(intervention_space(pomis_set))

    # Run KL-UCB over the resulting arm set
    return kl_ucb_fn(bandit_env, arms, horizon)
