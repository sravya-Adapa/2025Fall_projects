import networkx as nx
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from pathlib import Path


def visualize(G):
    """
    This function is used to visualize the Tesla supplier network graph in 3 layer layout.
    Suppliers -> Industries -> Tesla
    The visualization includes:
        • Node positions arranged vertically for each layer
        • Directed edges with arrows
        • Edge labels showing weights
        • A legend describing each layer
    :param G: NetworkX graph
    :return: None
    """
    plt.figure(figsize=(12, 10))
    color_suppliers = '#87CEFA'
    color_industries = '#90EE90'
    color_tesla = '#FFD700'

    layer_2_nodes = [n for n, d in G.in_degree() if d == 0]
    layer_1_nodes = [n for n in G.predecessors(root_node)]
    layer_0_nodes = [root_node]

    pos = {} # simple 3 column layout
    for i, node in enumerate(layer_2_nodes):
        x_pos = (i + 0.5) * (len(layer_1_nodes) / max(len(layer_2_nodes), 1))
        pos[node] = (0, x_pos)
        y_pos = (i + 0.5) * (len(layer_1_nodes) / max(len(layer_2_nodes), 1))
        pos[node] = (0, y_pos)

    for i, node in enumerate(layer_1_nodes):
        pos[node] = (1, i + 0.5)

    pos[root_node] = (2, len(layer_1_nodes) / 2)
    nx.draw_networkx_nodes(G, pos, nodelist=layer_2_nodes, node_color=color_suppliers, node_size=300, label="Suppliers")
    nx.draw_networkx_nodes(G, pos, nodelist=layer_1_nodes, node_color=color_industries, node_size=1500,
                           label="Industries")
    nx.draw_networkx_nodes(G, pos, nodelist=layer_0_nodes, node_color=color_tesla, node_size=3000, label="Tesla")
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in G.edges(data=True)
        if d.get('weight') is not None
    }

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    patch1 = mpatches.Patch(color=color_suppliers, label='Suppliers (Layer 2)')
    patch2 = mpatches.Patch(color=color_industries, label='Industries (Layer 1)')
    patch3 = mpatches.Patch(color=color_tesla, label='Tesla Product (Layer 0)')
    plt.legend(handles=[patch1, patch2, patch3], loc='lower right', fontsize=10, frameon=True, shadow=True)
    plt.title("Tesla Supply Chain Network", fontsize=16)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    industry_path = os.path.join(base_dir, "Tesla_specific_data","preprocessed_data", "component_weights_2024_percent.csv")
    supplier_path = os.path.join(base_dir, "Tesla_specific_data", "Supplier_to_BEA_commodity_mapping.csv")

    # Load the data
    df_industries = pd.read_csv(industry_path)
    df_suppliers = pd.read_csv(supplier_path)

    # Layer 0/ root node
    G = nx.DiGraph()
    root_node = "Tesla"
    G.add_node(root_node, layer=0)

    # Layer1(industries)
    for _, row in df_industries.iterrows():
        industry = row["component_industry_description"]
        weight = row["weight_share_percent"]
        G.add_edge(industry, root_node, weight=weight, layer=1)

    # Layer2(suppliers)
    layer2_weight = np.random.default_rng(42) # So that weights are reproducible

    for industry, grp in df_suppliers.groupby("BEA Commodity Row"):
        # only assign for industries present in the graph
        if not G.has_node(industry):
            continue
        suppliers = grp["Supplier"].tolist()
        n = len(suppliers)
        if n == 0:
            continue

        # random positive shares that sum to 100%
        shares_pct = layer2_weight.dirichlet(np.ones(n)) * 100.0

        for supplier, pct in zip(suppliers, shares_pct):
            G.add_edge(supplier, industry, weight=float(pct), layer=2)

    visualize(G)

    out_dir = Path(base_dir) / "supply_chain_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = out_dir / "tesla_supply_chain_graph_2024.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved graph to: {graph_path}")
    print(type(G), "nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())



