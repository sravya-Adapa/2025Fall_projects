import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os

# CONFIGURATION
NUM_SIMS = 10000 # number of monte carlo runs
P_FAIL_BASE = 0.05 # baseline supplier failure probability
TOTAL_COGS_REFERENCE = 80_240_000_000 # target COGS (USD) for the year 2024
COGS_PER_VEHICLE_USD =  45245.32  # COGS_PER_VEHICLE production value for the year 2024


def load_graph_from_pickle(file_path):
    """
    This function simply loads the saved network graph.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Graph file not found at: {file_path}")
    with open(file_path, 'rb') as f:
        G = pickle.load(f)
    return G


def load_industry_cogs():
    """
    Loads the preprocessed data file and extract the dictionary of cogs allocated value for each industries.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "Tesla_specific_data", "preprocessed_data", "component_weights_2024_percent.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    cogs_industry = df.set_index('component_industry_description')['cogs_allocated_usd'].to_dict()

    return cogs_industry


def simulation(G, cogs_map):
    """
    Main MC model design
    """
    # Find
    suppliers = [n for n, d in G.in_degree() if d == 0] # selecting supplier nodes
    if G.has_node("Tesla"): # root towards the Tesla node
        root = "Tesla"
    industries = [n for n in G.predecessors(root)] # industries nodes with edges into Tesla

    # Get Original COGS
    original_industry_cogs = {} # empty dictionary which contains COGS values for just industry that actually feed to root node.
    for ind in industries:
        original_industry_cogs[ind] = cogs_map[ind]

    # Results
    simulated_total_cogs_results = [] # empty list for storing per run simulated COGS value
    bottleneck_counts = {ind: 0 for ind in industries} # number of runs each industry was giving most limiting availability

    for _ in range(NUM_SIMS):

        # Monte Carlo loop

        # to determine which supplier nodes to fail
        is_failed = np.random.rand(len(suppliers)) < P_FAIL_BASE
        failed_indices = np.where(is_failed)[0] # recording which indices was failed

        # to determine the severity of failure. Starting with 1 for every industry and changing it bases of failure severity
        supplier_damage = np.zeros(len(suppliers)) # initially 0 means no damage for all suppliers
        if len(failed_indices) > 0:
            failure_severity = np.random.uniform(0.3, 1.0, size=len(failed_indices)) #For failed suppliers, samples a
            # fractional damage between 30% and 100%. (So a failed supplier doesn’t always lose 100% of their contribution; it
            # can be a partial outage.)
            supplier_damage[failed_indices] = failure_severity
        supplier_dict = dict(zip(suppliers, supplier_damage)) #mapping supplier node with the damage fraction

        # step 2 calculate remaining COGS %
        total_COGS_available = 0.0 # per run sum for simulated COGS
        remaining_industry_percentage = {} # empty dict to track each industry's surviving percentage

        for industry in industries: # This loop runs remaining share after supplier disruptions
            percentage_left = 1.0 #initialize with 100 percent
            for supplier, _, data in G.in_edges(industry, data=True): # for incoming supplier edge into that industry
                weight = data.get('weight', 0) # get the weights
                supplier_damage = supplier_dict.get(supplier, 0) # get the damage fraction value
                loss = weight * supplier_damage # reduces the available share
                percentage_left -= loss # final available share
            percentage_left = max(0.0, percentage_left)
            remaining_industry_percentage[industry] = percentage_left # it stores surviving fraction for that industry

            # Value Left = Original Dollar Amount * % Available
            value_left = original_industry_cogs[industry] * percentage_left # converts share into final COGS value
            total_COGS_available += value_left # summation across industries to figure out total impact
        simulated_total_cogs_results.append(total_COGS_available)

        # step -3 Find bottleneck industry
        # Identifies the most starved industry in this run (smallest surviving percentage).
        # If there’s any shortfall (<100%), increments that industry’s bottleneck counter.

        min_percentage = min(remaining_industry_percentage.values())
        if min_percentage < 1.0:
            bottleneck = [k for k, v in remaining_industry_percentage.items() if v == min_percentage]
            for b in bottleneck:
                bottleneck_counts[b] += 1

    return simulated_total_cogs_results, bottleneck_counts


if __name__ == "__main__":

    # Load paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "supply_chain_graph", "tesla_supply_chain_graph_2024.pkl")

    # Load network graph
    G = load_graph_from_pickle(graph_path)

    # Load cogs per industry
    industry_cogs = load_industry_cogs()

    # Run simulation
    simulation_cogs, bottlenecks = simulation(G, industry_cogs)
    avg_cogs_sum = np.mean(simulation_cogs)
    avg_units_from_cogs = avg_cogs_sum / COGS_PER_VEHICLE_USD
    reduction_sum = TOTAL_COGS_REFERENCE - avg_cogs_sum
    shortfalls = [TOTAL_COGS_REFERENCE - x for x in simulation_cogs]
    p95_shortfall = np.percentile(shortfalls, 95)

    # Print Results
    print(f"Total Baseline COGS:        ${TOTAL_COGS_REFERENCE / 1e9:.2f}B")
    print(f"Implied Avg Units (from simulated COGS): {int(round(avg_units_from_cogs)):,}")
    print(f"Units reduction vs baseline: {((1_773_443 - avg_units_from_cogs) / 1_773_443):.0%}")
    print(f"Simulated Avg Availability: ${avg_cogs_sum / 1e9:.2f}B")
    print(f"Average Shortfall:          ${reduction_sum / 1e9:.2f}B ({(reduction_sum / TOTAL_COGS_REFERENCE):.1%})")
    print(f"P95 Shortfall (Risk):       ${p95_shortfall / 1e9:.2f}B")
    sorted_bn = sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)
    for ind, count in sorted_bn[:5]:
        freq = (count / NUM_SIMS) * 100
        print(f"{ind}{' ' * (40 - len(str(ind)))} | {freq:.1f}% of runs")

