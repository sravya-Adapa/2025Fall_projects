import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

NUM_SIMULATIONS= 10000 # number of simulations
PROBABILITY_OF_FAILURE = 0.05 # the probability baseline for a node to fail.
#test value
total_cogs = 100000
total_units = 100

def build_network():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    industry_path = os.path.join(base_dir, "Tesla_specific_data", "BEA_industry_to_commodity_updated_data.csv")
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
        industry = row["Industry Description"]
        weight = row["Motor vehicles, bodies and trailers, and parts"]
        G.add_edge(industry, root_node, weight=weight, layer=1)

    # Layer2(suppliers)
    supplier_col = df_suppliers.columns[0]
    industry_col = df_suppliers.columns[1]
    grouped = df_suppliers.groupby(industry_col)[supplier_col].apply(list)
    for industry, suppliers in grouped.items():
        if G.has_node(industry):
            # randomized weight dependencies for suppliers to each industry
            n = len(suppliers)
            splits = np.random.rand(n)
            splits = splits / np.sum(splits)
            for supplier, split in zip(suppliers, splits):
                G.add_edge(supplier, industry, weight=split, layer=2)
    return G

def simulate(G,n_simulations=NUM_SIMULATIONS,p_failure=PROBABILITY_OF_FAILURE):
    # to store results for each simulation
    unit_results = []
    trapped_value_results = []
    suppliers = [n for n, d in G.in_degree() if d == 0] # suppliers lists from the graph if there is no incoming edge then it is considered as supplier
    industries = [n for n in G.predecessors("Tesla")] # industry list from graph
    for _ in range(n_simulations):
        # assign random failure probability to each supplier.
        # If the random failure probability is less than PROBABILITY_OF_FAILURE only then that supplier node will fail.
        supplier_fail_probability = np.random.rand(len(suppliers))
        fail = supplier_fail_probability < p_failure
        # list of failed suppliers
        failed_suppliers = [
            supplier for supplier, is_failed in zip(suppliers, fail)
            if is_failed
        ]

        # Calculate how much percent of supply each industry have after failing random supplier nodes.
        # starting with 100% supply if one fails subtract the weight of the edge from the total supply.
        # after subtracting all weights of failed nodes we get the final supply for each industry.
        industry_supply = {}
        for industry in industries:
            current_supply = 1
            for supplier,_,data in G.in_edges(industry, data=True):
                if supplier in failed_suppliers:
                    current_supply -= data["weight"]
            industry_supply[industry] = max(0, current_supply)

        # check the weakest industry. If batteries(industry) are left with 40% of original supply and all other industries have more than 40%
        # then Tesla’s production is limited to 40%, since overall output cannot exceed the weakest link in the supply chain.
        min_supply = min(industry_supply.values())

        # 1. Production loss interms of units
        loss_percentage = 1- min_supply
        units_loss = loss_percentage * total_units
        unit_results.append(loss_percentage)

        # 2. Trapped Value
        trapped_value = 0.0

        for industry, supply_level in industry_supply.items():
            excess_portion = supply_level - min_supply
            industry_weight = G[industry]["Tesla"]['weight']
            #Value = Excess% * Weight_of_Industry * Total_Annual_COGS
            value_stuck = excess_portion * industry_weight * total_cogs
            trapped_value += value_stuck

        trapped_value_results.append(trapped_value)
    return unit_results, trapped_value_results

if __name__ == "__main__":
    G = build_network()
    unit_results, trapped_value_results = simulate(G)
    avg_units = np.mean(unit_results)
    p95_units = np.percentile(unit_results, 95)

    avg_trapped = np.mean(trapped_value_results)
    p95_trapped = np.percentile(trapped_value_results, 95)

    print(f"PRODUCTION LOSS (UNITS)")
    print(f"Average: {avg_units:,.0f} vehicles")
    print(f"P95 Risk: {p95_units:,.0f} vehicles")

    print(f"TRAPPED INVENTORY VALUE ($)")
    print(f"Average: ${avg_trapped}")
    print(f"P95 Risk: ${p95_trapped}")















