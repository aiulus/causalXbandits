import networkx as nx
from algorithms.pomis import get_estimand_sets

def make_toy_scm_graph():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Z", "X"),
        ("X", "Y")
    ])
    # Mark confounding between Z and Y via bidirected edge (hidden confounder)
    G.add_edge("Z", "Y", confounded=True)
    return G


if __name__ == "__main__":
    G = make_toy_scm_graph()
    reward_variable = "Y"

    pomis_sets = get_estimand_sets(G, reward_variable)
    print("Identified POMIS sets:")
    for s in pomis_sets:
        print(f"  - {s}")
