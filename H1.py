import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from Simulation import*


# CONFIGURATION
NUM_SIMS = 10000
P_FAIL_BASE = 0.05
TOTAL_COGS_REFERENCE = 80_240_000_000

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


def H1_simulation(G, cogs_map):

    suppliers = [n for n, d in G.in_degree() if d == 0]
    root = "Tesla"
    industries = list(G.predecessors(root))
    original_industry_cogs = {ind: cogs_map[ind] for ind in industries}

    # Identify Top-2 industries by original COGS
    sorted_industries = sorted(
        original_industry_cogs.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_2_industries = [x[0] for x in sorted_industries[:2]]

    # Results
    simulated_cogs_sum = {ind: 0.0 for ind in industries}
    cogs_loss_sum = {ind: 0.0 for ind in industries}
    severity_sum = {ind: 0.0 for ind in industries}
    failure_event_count = {ind: 0 for ind in industries}
    bottleneck_counts = {ind: 0 for ind in industries}
    top2_loss_share_per_run = []

    for _ in range(NUM_SIMS):

        total_loss_this_run = 0.0
        top2_loss_this_run = 0.0
        remaining_industry_percentage = {}

        # Step -1
        is_failed = np.random.rand(len(suppliers)) < P_FAIL_BASE
        failed_indices = np.where(is_failed)[0]

        supplier_damage = np.zeros(len(suppliers))
        if len(failed_indices) > 0:
            supplier_damage[failed_indices] = np.random.uniform(
                0.3, 1.0, size=len(failed_indices)
            )

        supplier_damage_map = dict(zip(suppliers, supplier_damage))

        # step -2
        for industry in industries:
            pct_remaining = 1.0
            for supplier, _, data in G.in_edges(industry, data=True):
                weight = data.get("weight", 0)
                damage = supplier_damage_map.get(supplier, 0)
                pct_remaining -= weight * damage

            pct_remaining = max(0.0, pct_remaining)
            remaining_industry_percentage[industry] = pct_remaining

            # Compute COGS values
            original_cogs = original_industry_cogs[industry]
            industry_simulated_cogs = original_cogs * pct_remaining
            industry_cogs_loss = original_cogs - industry_simulated_cogs
            simulated_cogs_sum[industry] += industry_simulated_cogs
            cogs_loss_sum[industry] += industry_cogs_loss

            # Track severity + failure count
            if pct_remaining < 1.0:
                severity_sum[industry] += (1 - pct_remaining)
                failure_event_count[industry] += 1

            # Run-level loss (for Top-2 share)
            total_loss_this_run += industry_cogs_loss
            if industry in top_2_industries:
                top2_loss_this_run += industry_cogs_loss

        # step -3
        min_pct = min(remaining_industry_percentage.values())
        if min_pct < 1.0:
            for ind, pct in remaining_industry_percentage.items():
                if pct == min_pct:
                    bottleneck_counts[ind] += 1


        if total_loss_this_run > 0:
            top2_loss_share_per_run.append(top2_loss_this_run / total_loss_this_run)
        else:
            top2_loss_share_per_run.append(0.0)

    # Final output dict
    stats_out = {}
    for industry in industries:
        failures = failure_event_count[industry]
        avg_severity = severity_sum[industry] / failures if failures > 0 else 0.0
        stats_out[industry] = {
            "original_cogs": original_industry_cogs[industry],
            "avg_simulated_cogs": simulated_cogs_sum[industry] / NUM_SIMS,
            "avg_loss": cogs_loss_sum[industry] / NUM_SIMS,
            "avg_severity": avg_severity,
            "bottleneck_freq": bottleneck_counts[industry],
            "failure_events": failures,
        }

    return stats_out, top2_loss_share_per_run, top_2_industries



def plot_results(stats_data, loss_shares, top_2_names):
    sorted_inds = sorted(stats_data.keys(), key=lambda k: stats_data[k]["original_cogs"], reverse=True)
    avg_losses = [stats_data[k]["avg_loss"] / 1e9 for k in sorted_inds]
    labels = sorted_inds
    colors = ['firebrick' if x in top_2_names else 'steelblue' for x in labels]

    loss_shares_pct = [x * 100 for x in loss_shares]
    mean_share_pct = np.mean(loss_shares_pct)

    plt.figure(figsize=(8, 6))
    plt.hist(loss_shares_pct, bins=50, color='gray', alpha=0.7)
    plt.axvline(50, color='black', linestyle='--', linewidth=2, label='Hypothesis Threshold (50%)')
    plt.axvline(mean_share_pct, color='red', linewidth=2, label=f'Actual Mean ({mean_share_pct:.1f}%)')
    plt.title("Distribution of Top 2 Categories' Contribution to Loss", fontsize=14)
    plt.xlabel("Share of Total Loss (0–100%)", fontsize=12)
    plt.ylabel("Number of Simulations", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Force y-axis to show full 10,000 simulations
    plt.ylim(0, len(loss_shares))

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, avg_losses, color=colors)
    plt.yticks(y_pos, labels, fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlabel("Average Financial Loss ($ Billions)", fontsize=12)
    plt.title("Financial Loss Drivers by Category", fontsize=14)

    top_loss_sum = sum(stats_data[t]["avg_loss"] for t in top_2_names)
    total_loss_sum = sum(d["avg_loss"] for d in stats_data.values())
    top_share_pct = (top_loss_sum / total_loss_sum) * 100

    plt.text(0.5, 0.95, f"Top 2 Share: {top_share_pct:.1f}%",
             transform=plt.gca().transAxes, ha='center', va='top',
             bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "supply_chain_graph", "tesla_supply_chain_graph_2024.pkl")
    G = load_graph_from_pickle(graph_path)
    industry_cogs = load_industry_cogs()
    stats_data, loss_shares, top_2_names = H1_simulation(G, industry_cogs)

    # --- REPORTING ---
    total_original = sum(d["original_cogs"] for d in stats_data.values())
    total_simulated = sum(d["avg_simulated_cogs"] for d in stats_data.values())
    total_loss = sum(d["avg_loss"] for d in stats_data.values())
    sorted_inds = sorted(stats_data.keys(), key=lambda k: stats_data[k]["original_cogs"], reverse=True)


    print(f"Target Threshold (Hypothesis H1): > 50% of Total Loss driven by Top 2 Categories")
    header = f"{'Category':<45} | {'Orig COGS ($B)':<14} | {'Avg Loss ($B)':<14} | {'Freq (BN)':<10} | {'Avg Severity':<12}"
    print(header)
    for ind in sorted_inds:
        d = stats_data[ind]
        orig = d["original_cogs"] / 1e9
        loss = d["avg_loss"] / 1e9
        freq_pct = (d["bottleneck_freq"] / NUM_SIMS) * 100
        sev_pct = d["avg_severity"] * 100
        print(f"{ind:<45} | ${orig:<13.2f} | ${loss:<13.3f} | {freq_pct:<9.1f}% | {sev_pct:.1f}%")
    print(f"Original Total COGS:                ${total_original / 1e9:,.2f} B")
    print(f"Simulated Average Total COGS:       ${total_simulated / 1e9:,.2f} B")
    print(f"Average Total Loss (All Combined):  ${total_loss / 1e9:,.2f} B")

    top_1 = sorted_inds[0]
    top_2 = sorted_inds[1]
    loss_1 = stats_data[top_1]["avg_loss"]
    loss_2 = stats_data[top_2]["avg_loss"]
    share_1 = (loss_1 / total_loss) * 100
    share_2 = (loss_2 / total_loss) * 100
    combined_share = share_1 + share_2

    print(f"Contribution of Top Categories to Total Loss:")
    print(f"   - {top_1}: {share_1:.1f}%")
    print(f"   - {top_2}: {share_2:.1f}%")
    print(f"   - Combined Top 2 Contribution: {combined_share:.1f}%")

    t_stat, p_value = stats.ttest_1samp(loss_shares, popmean=0.5, alternative='greater')
    print(f"STATISTICAL VERDICT (T-Test):")
    print(f"   T-Statistic: {t_stat:.4f} | P-Value: {p_value:.4e}")
    if p_value < 0.05 and t_stat > 0:
        print("H1 SUPPORTED: Top 2 categories significantly drive >50% of losses.")
    else:
        print("H1 REJECTED: Top 2 categories do not significantly drive >50% of losses.")


    # --- GENERATE PLOTS ---
    plot_results(stats_data, loss_shares, top_2_names)