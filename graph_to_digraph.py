import os
import pickle
import networkx as nx

def convert_to_digraph(src_path: str, dst_path: str | None = None, root_node: str = "Tesla") -> nx.DiGraph:
    """
    Read a (possibly undirected/multi) graph from `src_path`, build a directed version with
    edges oriented from supplier to industry and industry to Tesla (preserving 'weight'), and write
    it to `dst_path` (default: "<src>_directed.pkl"). Returns the directed graph.
    """
    with open(src_path, "rb") as f:
        G0 = pickle.load(f)

    # Choose output path
    if dst_path is None:
        base, ext = os.path.splitext(src_path)
        dst_path = f"{base}_directed{ext}"

    # If already DiGraph, just copy to new file and return
    if isinstance(G0, nx.DiGraph):
        with open(dst_path, "wb") as f:
            pickle.dump(G0, f, protocol=pickle.HIGHEST_PROTOCOL)
        return G0

    if root_node not in G0:
        raise ValueError(f"Root node '{root_node}' not found in the graph at {src_path}")

    # Industries are nodes directly connected to Tesla
    industries = set(G0.predecessors(root_node)) if G0.is_directed() else set(G0.neighbors(root_node))

    # Suppliers are neighbors/predecessors of industries that are neither Tesla nor other industries
    suppliers = set()
    for ind in industries:
        nbrs = set(G0.predecessors(ind)) if G0.is_directed() else set(G0.neighbors(ind))
        suppliers.update(n for n in nbrs if n != root_node and n not in industries)

    D = nx.DiGraph()

    # Copy node attributes
    for n, attrs in G0.nodes(data=True):
        D.add_node(n, **(attrs or {}))

    def _edge_weight(u, v) -> float:
        """
        Robustly read 'weight' from Graph/MultiGraph/DiGraph in either direction if needed.
        Sums parallel edges if present (MultiGraph).
        """
        if G0.has_edge(u, v):
            data = G0.get_edge_data(u, v, default={})
        elif not G0.is_directed() and G0.has_edge(v, u):
            data = G0.get_edge_data(v, u, default={})
        else:
            data = {}

        # MultiGraph returns mapping: {key: {attr...}}; Graph returns {attr...}
        if isinstance(data, dict) and data and any(isinstance(val, dict) for val in data.values()):
            return float(sum((val or {}).get("weight", 0.0) for val in data.values()))
        return float(data.get("weight", 0.0)) if isinstance(data, dict) else 0.0

    # Orient industry -> Tesla
    for ind in industries:
        w = _edge_weight(ind, root_node)
        D.add_edge(ind, root_node, weight=w)

    # Orient supplier -> industry
    for ind in industries:
        nbrs = set(G0.predecessors(ind)) if G0.is_directed() else set(G0.neighbors(ind))
        for sup in nbrs:
            if sup in suppliers:
                w = _edge_weight(sup, ind)
                D.add_edge(sup, ind, weight=w)

    # Write directed copy
    with open(dst_path, "wb") as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)

    return D


if __name__ == "__main__":
    # Load paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_graph_path = os.path.join(base_dir, "supply_chain_graph", "tesla_supply_chain_graph_2024.pkl")
    dst_graph_path = os.path.join(base_dir, "supply_chain_graph", "tesla_supply_chain_graph_2024_directed.pkl")

    # Load and convert network graph (saving as a new file)
    G = convert_to_digraph(src_graph_path, dst_path=dst_graph_path)

    print(f"Source graph:   {src_graph_path}")
    print(f"Directed copy:  {dst_graph_path}")
    print(type(G), "nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
