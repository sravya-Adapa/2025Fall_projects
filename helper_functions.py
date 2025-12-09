# Import libraries needed for all the hypothesis testing

import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("TkAgg")
from typing import Dict,Tuple,List,Optional
import math
import matplotlib.pyplot as plt


# CONFIGURATION
NUM_SIMS = 10000 # number of monte carlo runs
P_FAIL_BASE = 0.05 # baseline supplier failure probability
TOTAL_COGS_REFERENCE = 80_240_000_000 # target COGS (USD) for the year 2024
COGS_PER_VEHICLE_USD =  45245.32  # COGS_PER_VEHICLE production value for the year 2024


def load_graph_from_pickle(file_path):
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

def load_cogs_per_industry():
    """
    Loads the preprocessed data file and extract the dictionary of cogs allocated value for each industries.
    :param: None
    :return: dictionary of industry cogs values for each industry
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Tesla_specific_data", "preprocessed_data", "component_weights_2024_percent.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    cogs_industry = df.set_index('component_industry_description')['cogs_allocated_usd'].to_dict()

    return cogs_industry

def get_top2_industries(original_industry_cogs):
    sorted_industries = sorted(original_industry_cogs.items(),
                               key=lambda x: x[1],
                               reverse=True)
    return [x[0] for x in sorted_industries[:2]]

def _w01(x: float) -> float:
    """Coerce an edge weight into [0,1] even if stored as a percent (e.g., 23.5 -> 0.235)."""
    return x / 100.0 if x > 1.0 else x

def get_industry_suppliers_and_shares(G: nx.DiGraph, industry: str) -> Dict[str, float]:
    """
    Return supplier share vector feeding *into* an industry (DIRECTED):
      shares = {supplier: share_in_[0,1]}, normalized to sum to 1.
    Uses G.in_edges(industry, data=True) so direction matters.
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


