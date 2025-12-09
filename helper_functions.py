# Import libraries needed for all the hypothesis testing
import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("TkAgg")
from typing import Dict,Tuple,List,Optional,Any
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt, erf
from scipy.stats import norm
from statistics import NormalDist
import matplotlib.patches as mpatches




# CONFIGURATION
NUM_SIMS = 10000 # number of monte carlo runs
P_FAIL_BASE = 0.05 # baseline supplier failure probability
TOTAL_COGS_REFERENCE = 80_240_000_000 # target COGS (USD) for the year 2024
COGS_PER_VEHICLE_USD =  45245.32  # COGS_PER_VEHICLE production value for the year 2024


def load_graph_from_pickle(file_path: str) -> nx.Graph:
    """
    This function simply loads the saved network graph.
    :param file_path: path to the pickle file
    :return: networkx graph

    >>> load_graph_from_pickle("non_existing_file.pkl")  # doctest: +IGNORE_EXCEPTION_DETAIL +ELLIPSIS
    Traceback (most recent call last):
    FileNotFoundError: ...


    >>> import tempfile, pickle, networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edge("A", "B", weight=0.8)
    >>> temp = tempfile.NamedTemporaryFile(delete=False)
    >>> _ = pickle.dump(G, open(temp.name, "wb"))
    >>> loaded = load_graph_from_pickle(temp.name)
    >>> isinstance(loaded, nx.DiGraph)
    True
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Graph file not found at: {file_path}")
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    return G

def load_cogs_per_industry()-> Dict[str, float]:
    """
    Loads the preprocessed data file and extract the dictionary of cogs value for each industry.
    :param: None
    :return: dictionary of industry cogs values for each industry

    >>> import pandas as pd
    >>> import types
    >>> fake_df = pd.DataFrame({
    ...     "component_industry_description": ["A", "B"],
    ...     "cogs_allocated_usd": [10, 20]
    ... })
    >>> old_read_csv = pd.read_csv
    >>> pd.read_csv = lambda path: fake_df
    >>> result = load_cogs_per_industry()
    >>> sorted(result.items())
    [('A', 10), ('B', 20)]
    >>>
    >>> pd.read_csv = old_read_csv
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Tesla_specific_data", "preprocessed_data", "component_weights_2024_percent.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    cogs_industry = df.set_index('component_industry_description')['cogs_allocated_usd'].to_dict()

    return cogs_industry

def get_top2_industries(original_industry_cogs):
    """
    This function extracts the top 2 industries based on COGS distribution from the original industry cogs dictionary.
    :param original_industry_cogs: dictionary of industry cogs values
    :return: list of top 2 industries

    >>> get_top2_industries({"A": 100, "B": 300, "C": 200})
    ['B', 'C']

    >>> get_top2_industries({"A": 500})
    ['A']

    >>> get_top2_industries({})
    []

    """
    sorted_industries = sorted(original_industry_cogs.items(),
                               key=lambda x: x[1],
                               reverse=True)
    return [x[0] for x in sorted_industries[:2]]

def _w01(x: float) -> float:
    """Coerce an edge weight into [0,1] even if stored as a percent (e.g., 23.5 -> 0.235).
    :param x: edge weight
    :return: normalized edge weight

    >>> _w01(0.5)
    0.5

    >>> _w01(5)
    0.05

    >>> _w01(0)
    0
    """

    return x / 100.0 if x > 1.0 else x

def get_industry_suppliers_and_shares(G: nx.DiGraph, industry: str) -> Dict[str, float]:
    """
    Return supplier share vector feeding *into* an industry (DIRECTED):
      shares = {supplier: share_in_[0,1]}, normalized to sum to 1.
    Uses G.in_edges(industry, data=True) so direction matters.
    :param G: networkx DiGraph
    :param industry: string
    :return: dictionary of industry shares

    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edge("S1", "I1", weight=0.2)
    >>> G.add_edge("S2", "I1", weight=0.3)
    >>> G.add_edge("S3", "I1", weight=0.5)
    >>> out = get_industry_suppliers_and_shares(G, "I1")
    >>> sorted((k, round(v, 3)) for k, v in out.items())
    [('S1', 0.2), ('S2', 0.3), ('S3', 0.5)]

    >>> G = nx.DiGraph()
    >>> G.add_edge("S1", "I1", weight=20)   # becomes 0.20
    >>> G.add_edge("S2", "I1", weight=30)   # becomes 0.30
    >>> out = get_industry_suppliers_and_shares(G, "I1")
    >>> sorted((k, round(v, 3)) for k, v in out.items())
    [('S1', 0.4), ('S2', 0.6)]

     >>> G = nx.DiGraph()
    >>> get_industry_suppliers_and_shares(G, "")
    {}
    """

    pairs = []
    for sup, _, d in G.in_edges(industry, data=True):   # suppliers → industry
        w = _w01(float(d.get("weight", 0.0)))
        if w > 0.0:
            pairs.append((sup, w))
    if not pairs:
        return {}
    total = sum(w for _, w in pairs)
    if total <= 0.0:
        return {}
    # Normalize so representation (percent vs fraction) and drift don’t affect logic
    return {sup: w / total for sup, w in pairs}

def build_independent_draws(
    G: nx.DiGraph,
    num_runs: int,
    p_fail: float,
    seed: int = 123,
    severity_low: float = 0.3,
    severity_high: float = 1.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    For each supplier (in-degree==0), precompute:
      - Bernoulli failure flags (unused here but kept for clarity)
      - severity array s_i[r] in [0,1] where non-failures have 0, failures ~ U[low, high]
    Returned as: {supplier: (fails_bool, severities_float)}

    >>> G = nx.DiGraph()
    >>> G.add_edge("S1", "I1")
    >>> G.add_edge("S2", "I1")
    >>> num_runs = 5
    >>> out = build_independent_draws(G, num_runs=num_runs, p_fail=0.5, seed=0)
    >>> sorted(out.keys())
    ['S1', 'S2']

    >>> fails_S1, sev_S1 = out["S1"]
    >>> lo, hi = 0.3, 1.0
    >>> all((sev_S1[fails_S1] >= lo) & (sev_S1[fails_S1] <= hi))
    True
    >>> all(sev_S1[~fails_S1] == 0)
    True
    """
    rng = np.random.default_rng(seed)
    suppliers = [n for n, d in G.in_degree() if d == 0]
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for s in suppliers:
        fails = rng.random(num_runs) < p_fail
        sev = np.zeros(num_runs, dtype=np.float32)
        if fails.any():
            sev[fails] = rng.uniform(severity_low, severity_high, size=fails.sum()).astype(np.float32)
        out[s] = (fails, sev)
    return out


