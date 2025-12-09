from helper_functions import *
from typing import Dict, Tuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

"""
Hypothesis 1 proposes that within Tesla’s supply chain, the two industries with the highest baseline cost contributions are also the dominant drivers of loss when random supplier failures occur. 
In other words, although many upstream industries feed into Tesla, the expectation is that only the top two account for more than half of the total loss across all simulated disruptions. 
If this hypothesis is true, it would indicate a highly concentrated vulnerability in Tesla’s supply chain.
a Monte Carlo simulation is built that runs 10,000, in each simulated scenario, random suppliers are selected to fail.
and each supplier failure severity is also randomly assigned between 30% and 100%. 
For each run, each industry loses part of its COGS depending on the severity and weight of its failing suppliers.
As the simulation runs repeatedly, it calculates how much each industry contributes to total loss, how often each one becomes a bottleneck,
how much total loss occurred, and how much of that loss came specifically from the top two industries.
Then we test the hypothesis using one-sample t test that compares the observed mean loss share(top 2 industries) to the hypothesized threshold of 0.5
"""


def simulation(G: nx.DiGraph, cogs_map: dict):
    """
    This is the Simulation model.
    Step 1: Randomly select suppliers to fail and determine severity of failure
    Step 2: Run the simulation for 10000 times and collect the results[for each run identify the loss caused by failure nodes]
    Step 3: Identify the bottleneck industries for each run
    """
    suppliers = [n for n, d in G.in_degree() if d == 0]
    root = "Tesla"
    industries = list(G.predecessors(root))
    original_industry_cogs = {ind: cogs_map[ind] for ind in industries}

    # Top-2 industries
    top_2_industries = get_top2_industries(original_industry_cogs)

    # Pre-generate all failure/severity random draws (your function)
    draws = build_independent_draws(
        G,
        num_runs=NUM_SIMS,
        p_fail=P_FAIL_BASE,
        seed=123
    )

    # BASELINE OUTPUTS
    simulated_total_cogs_results = []
    bottleneck_counts = {ind: 0 for ind in industries}

    # H1 OUTPUTS
    simulated_cogs_sum = {ind: 0.0 for ind in industries}
    cogs_loss_sum = {ind: 0.0 for ind in industries}
    severity_sum = {ind: 0.0 for ind in industries}
    failure_event_count = {ind: 0 for ind in industries}
    top2_loss_share_per_run = []

    # MONTE CARLO LOOP
    for r in range(NUM_SIMS):

        remaining_industry_percentage = {}
        total_cogs_available_this_run = 0.0
        total_loss_this_run = 0.0
        top_2_loss_this_run = 0.0

        for industry in industries:
            pct_remaining = 1.0

            # LOOP over suppliers
            for supplier, _, data in G.in_edges(industry, data=True):
                w = data.get("weight", 0)

                # YOUR random draw replacement
                fails, sev = draws[supplier]
                damage = sev[r]     # either 0 or severity in [0.3, 1.0]

                # Reduce % availability
                pct_remaining -= w * damage

            pct_remaining = max(0.0, pct_remaining)
            remaining_industry_percentage[industry] = pct_remaining

            # Convert to dollars
            orig = original_industry_cogs[industry]
            simulated_cogs = orig * pct_remaining
            loss = orig - simulated_cogs

            total_cogs_available_this_run += simulated_cogs

            # --- H1 Metrics ---
            simulated_cogs_sum[industry] += simulated_cogs
            cogs_loss_sum[industry] += loss

            if pct_remaining < 1:
                severity_sum[industry] += (1 - pct_remaining)
                failure_event_count[industry] += 1

            total_loss_this_run += loss

            if industry in top_2_industries:
                top_2_loss_this_run += loss

        # End industry loop

        simulated_total_cogs_results.append(total_cogs_available_this_run)

        # Determine bottleneck
        min_pct = min(remaining_industry_percentage.values())
        if min_pct < 1:
            for ind, pct in remaining_industry_percentage.items():
                if pct == min_pct:
                    bottleneck_counts[ind] += 1

        # Compute share of loss from top 2
        if total_loss_this_run > 0:
            top2_loss_share_per_run.append(top_2_loss_this_run / total_loss_this_run)
        else:
            top2_loss_share_per_run.append(0.0)

    # Final summary table
    stats_out = {}
    for ind in industries:
        failures = failure_event_count[ind]
        avg_severity = severity_sum[ind] / failures if failures > 0 else 0.0

        stats_out[ind] = {
            "original_cogs": original_industry_cogs[ind],
            "avg_simulated_cogs": simulated_cogs_sum[ind] / NUM_SIMS,
            "avg_loss": cogs_loss_sum[ind] / NUM_SIMS,
            "avg_severity": avg_severity,
            "bottleneck_freq": bottleneck_counts[ind],
            "failure_events": failures,
        }

    return (
        simulated_total_cogs_results,
        bottleneck_counts,
        stats_out,
        top2_loss_share_per_run,
        top_2_industries
    )



def print_h1_results(stats_data, loss_shares, top_2_names):

    total_loss = sum(d["avg_loss"] for d in stats_data.values())
    sorted_inds = sorted(stats_data.keys(),
                         key=lambda k: stats_data[k]["original_cogs"],
                         reverse=True)

    print("\n================ H1 HYPOTHESIS TESTING ======================\n")
    print("H1: Top 2 industries contribute more than 50% of total loss.\n")

    header = (
        f"{'Category':<45} | {'Orig COGS ($B)':<14} | "
        f"{'Avg Loss ($B)':<14} | {'Freq (BN)':<9} | {'Avg Severity':<12}"
    )
    print(header)

    for ind in sorted_inds:
        d = stats_data[ind]
        orig = d["original_cogs"] / 1e9
        loss = d["avg_loss"] / 1e9
        freq_pct = (d["bottleneck_freq"] / NUM_SIMS) * 100
        sev_pct = d["avg_severity"] * 100

        print(f"{ind:<45} | ${orig:<13.2f} | ${loss:<13.3f} | "
              f"{freq_pct:<8.1f}% | {sev_pct:.1f}%")

    # Top-2 contribution
    top_1, top_2 = top_2_names
    loss_1 = stats_data[top_1]["avg_loss"]
    loss_2 = stats_data[top_2]["avg_loss"]
    share_1 = (loss_1 / total_loss) * 100
    share_2 = (loss_2 / total_loss) * 100
    combined_share = share_1 + share_2

    print("\nContribution of Top 2:")
    print(f"  - {top_1}: {share_1:.1f}%")
    print(f"  - {top_2}: {share_2:.1f}%")
    print(f"  - Combined: {combined_share:.1f}%")

    # T-test
    t_stat, p_value = stats.ttest_1samp(loss_shares, popmean=0.5, alternative='greater')

    print("\nT-Test:")
    print(f"  T-Statistic: {t_stat:.4f}")
    print(f"  P-Value:     {p_value:.4e}")

    if p_value < 0.05 and t_stat > 0:
        print("\nH1 SUPPORTED")
    else:
        print("\nH1 DOES NOT SUPPORT.")


def plot_h1_results(stats_data, loss_shares, top_2_names):

    sorted_inds = sorted(stats_data.keys(),
                         key=lambda k: stats_data[k]["original_cogs"],
                         reverse=True)
    avg_losses = [stats_data[k]["avg_loss"] / 1e9 for k in sorted_inds]
    labels = sorted_inds
    colors = ['firebrick' if x in top_2_names else 'steelblue' for x in labels]

    # ---- Histogram ----
    loss_shares_pct = [x * 100 for x in loss_shares]
    mean_share_pct = np.mean(loss_shares_pct)

    plt.figure(figsize=(8, 6))
    plt.hist(loss_shares_pct, bins=50, color='gray', alpha=0.7)

    plt.axvline(50, color='black', linestyle='--', linewidth=2,
                label='Hypothesis Threshold (50%)')
    plt.axvline(mean_share_pct, color='red', linewidth=2,
                label=f'Actual Mean ({mean_share_pct:.1f}%)')

    plt.title("Distribution of Top-2 Categories' Contribution to Loss", fontsize=14)
    plt.xlabel("Share of Total Loss (0–100%)")
    plt.ylabel("Number of Simulations")
    plt.ylim(0, len(loss_shares))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- Bar Chart ----
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(labels))

    plt.barh(y_pos, avg_losses, color=colors)
    plt.yticks(y_pos, labels, fontsize=10)
    plt.gca().invert_yaxis()

    plt.xlabel("Average Financial Loss ($ Billions)")
    plt.title("Financial Loss Drivers by Category")

    total_loss = sum(d["avg_loss"] for d in stats_data.values())
    top_loss_sum = sum(stats_data[t]["avg_loss"] for t in top_2_names)
    share_pct = (top_loss_sum / total_loss) * 100
    red_patch = mpatches.Patch(color='firebrick', label='Top 2 Industries')
    blue_patch = mpatches.Patch(color='steelblue', label='Other Industries')
    plt.legend(handles=[red_patch, blue_patch], loc='lower right')

    plt.text(
        0.5, 0.95, f"Top 2 Share: {share_pct:.1f}%",
        transform=plt.gca().transAxes,
        ha='center', va='top',
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9)
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "supply_chain_graph",
                              "tesla_supply_chain_graph_2024.pkl")

    G = load_graph_from_pickle(graph_path)
    industry_cogs = load_cogs_per_industry()

    (
        simulation_cogs,
        bottlenecks,
        stats_data,
        loss_shares,
        top_2_names
    ) = simulation(G, industry_cogs)

    print("\n================ BASELINE SIMULATION RESULTS ================\n")
    avg_cogs_sum = np.mean(simulation_cogs)
    reduction_sum = TOTAL_COGS_REFERENCE - avg_cogs_sum
    shortfalls = [TOTAL_COGS_REFERENCE - x for x in simulation_cogs]
    p95_shortfall = np.percentile(shortfalls, 95)

    print(f"Total Baseline COGS:        ${TOTAL_COGS_REFERENCE/1e9:.2f}B")
    print(f"Simulated Avg Availability: ${avg_cogs_sum/1e9:.2f}B")
    print(f"Average Shortfall:          ${reduction_sum/1e9:.2f}B")
    print(f"P95 Shortfall (Risk):       ${p95_shortfall/1e9:.2f}B")

    sorted_bn = sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)
    print("\nTop Bottleneck Industries:")
    for ind, count in sorted_bn[:5]:
        freq = (count / NUM_SIMS) * 100
        print(f"{ind:40} | {freq:.1f}%")


    print_h1_results(stats_data, loss_shares, top_2_names)
    plot_h1_results(stats_data, loss_shares, top_2_names)
