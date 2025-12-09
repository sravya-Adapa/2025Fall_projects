from helper_functions import *
from scipy import stats

# CONFIGURATION
NUM_SIMS = 10000 # number of monte carlo runs
P_FAIL_BASE = 0.05 # baseline supplier failure probability
TOTAL_COGS_REFERENCE = 80_240_000_000 # target COGS (USD) for the year 2024
COGS_PER_VEHICLE_USD =  45245.32  # COGS_PER_VEHICLE production value for the year 2024

MODE = "live_demo" # There is an option to change to "live_demo" , "sampled_video"
DEMO_FRAMES = 400  # number of frames to animate in live_demo
SAMPLE_STRIDE = 10 # animate every k-th run in sampled_video
RUN_ANIMATION = True


def simulation(G, cogs_map):
    """
    This is the Simulation model.
    Step 1: Randomply select suppliers to fail and determine severity of failure
    Step 2: Run the simulation for 10000 times and collect the results[for each run identify the loss caused by failure nodes]
    Step 3: Identify the bottleneck industries for each run
    :param: G: networkx graph
    :return:
    """
    # identify suppliers, industries,
    suppliers = [n for n, d in G.in_degree() if d == 0]
    root = "Tesla"
    industries = list(G.predecessors(root))
    original_industry_cogs = {ind: cogs_map[ind] for ind in industries}

    # Identify Top-2 industries for hypothesis testing
    top_2_industries = get_top2_industries(original_industry_cogs)

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
    for _ in range(NUM_SIMS):

        supplier_damage = np.zeros(len(suppliers))
        is_failed = np.random.rand(len(suppliers)) < P_FAIL_BASE
        failed_indices = np.where(is_failed)[0]
        if len(failed_indices) > 0:
            supplier_damage[failed_indices] = np.random.uniform(
                0.3, 1.0, size=len(failed_indices)
            )
        supplier_damage_map = dict(zip(suppliers, supplier_damage))

        remaining_industry_percentage = {}
        total_cogs_available_this_run = 0.0
        total_loss_this_run = 0.0
        top_2_loss_this_run = 0.0

        for industry in industries:
            pct_remaining = 1.0
            for supplier, _, data in G.in_edges(industry, data=True):
                w = data.get("weight", 0)
                dmg = supplier_damage_map.get(supplier, 0)
                pct_remaining -= w * dmg

            pct_remaining = max(0.0, pct_remaining)
            remaining_industry_percentage[industry] = pct_remaining
            original_cogs = original_industry_cogs[industry]
            simulated_cogs = original_cogs * pct_remaining
            loss = original_cogs - simulated_cogs
            # Baseline
            total_cogs_available_this_run += simulated_cogs
            # H1 metrics
            simulated_cogs_sum[industry] += simulated_cogs
            cogs_loss_sum[industry] += loss
            if pct_remaining < 1:
                severity_sum[industry] += (1 - pct_remaining)
                failure_event_count[industry] += 1
            total_loss_this_run += loss
            if industry in top_2_industries:
                top_2_loss_this_run += loss
        simulated_total_cogs_results.append(total_cogs_available_this_run)

        # Bottleneck
        min_pct = min(remaining_industry_percentage.values())
        if min_pct < 1:
            for ind, pct in remaining_industry_percentage.items():
                if pct == min_pct:
                    bottleneck_counts[ind] += 1

        # H1 loss-share per run
        if total_loss_this_run > 0:
            top2_loss_share_per_run.append(top_2_loss_this_run / total_loss_this_run)
        else:
            top2_loss_share_per_run.append(0.0)

    # Final H1 statistics table
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

    # ---------------- H1 RESULTS ----------------
    print_h1_results(stats_data, loss_shares, top_2_names)

    # ---------------- H1 PLOTS ----------------
    plot_h1_results(stats_data, loss_shares, top_2_names)

    PRECOMP = precompute_runs(G, industry_cogs, total_runs=10_000, p_fail=P_FAIL_BASE, root='Tesla', seed=1)

    ANIM, FIG = animate_mc_snapshots(
        G, industry_cogs,
        milestones=range(500, 10001, 500),
        pause_ms=10_000,
        p_fail=P_FAIL_BASE,
        cogs_per_vehicle=COGS_PER_VEHICLE_USD,
        root_node="Tesla",
        precomp=PRECOMP,  # <<< sync!
        baseline_units=1_773_443,
    )
    plt.tight_layout()
    plt.show()
