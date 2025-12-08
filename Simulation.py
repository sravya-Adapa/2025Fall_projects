import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("TkAgg")   # place BEFORE importing pyplot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import time



# CONFIGURATION
NUM_SIMS = 10000 # number of monte carlo runs
P_FAIL_BASE = 0.05 # baseline supplier failure probability
TOTAL_COGS_REFERENCE = 80_240_000_000 # target COGS (USD) for the year 2024
COGS_PER_VEHICLE_USD =  45245.32  # COGS_PER_VEHICLE production value for the year 2024

# CONFIGURATION FOR ANIMATED PLOT
MODE = "live_demo" # There is an option to change to "live_demo" , "sampled_video"
DEMO_FRAMES = 400  # number of frames to animate in live_demo
SAMPLE_STRIDE = 50 # animate every k-th run in sampled_video
RUN_ANIMATION = True  # set False if there is no need for the animation window


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
    MC model design
    :param G:
    :param cogs_map:
    :return:
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
        loss_attribution = {ind: 0.0 for ind in industries}

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

        # step 3 Find bottleneck industry
        # Identifies the most starved industry in this run (smallest surviving percentage).
        # If there’s any shortfall (<100%), increments that industry’s bottleneck counter.

        min_percentage = min(remaining_industry_percentage.values())
        if min_percentage < 1.0:
            bottleneck = [k for k, v in remaining_industry_percentage.items() if v == min_percentage]
            for b in bottleneck:
                bottleneck_counts[b] += 1

    return simulated_total_cogs_results, bottleneck_counts

# Final animations functions

def precompute_runs(G, cogs_map, total_runs=10_000, p_fail=P_FAIL_BASE, root='Tesla', seed=1):
    suppliers = [n for n, d in G.in_degree() if d == 0]
    industries = [n for n in G.predecessors(root)]
    sup_idx = {s:i for i, s in enumerate(suppliers)}
    rng = np.random.default_rng(seed)

    n_sup = len(suppliers)
    # per-run per-supplier severity (0 means no failure)
    damage = np.zeros((total_runs, n_sup), dtype=np.float32)
    fails = rng.random((total_runs, n_sup)) < p_fail
    damage[fails] = rng.uniform(0.3, 1.0, size=fails.sum())

    # cache supplier->industry lists with weights (using supplier index)
    inlists = {
        ind: [(sup_idx[sup], float(data.get('weight', 0.0)))
              for sup, _, data in G.in_edges(ind, data=True) if sup in sup_idx]
        for ind in industries
    }
    orig = {ind: float(cogs_map[ind]) for ind in industries}

    all_cogs   = np.empty(total_runs, dtype=np.float64)
    bneck_idx  = np.empty(total_runs, dtype=np.int32)

    for r in range(total_runs):
        tot = 0.0
        min_pct, min_j = 1.1, 0
        row = damage[r]
        for j, ind in enumerate(industries):
            pct_left = 1.0
            for si, w in inlists[ind]:
                pct_left -= w * row[si]
            pct_left = max(0.0, pct_left)
            if pct_left < min_pct:
                min_pct, min_j = pct_left, j
            tot += orig[ind] * pct_left
        all_cogs[r] = tot
        bneck_idx[r] = min_j

    return suppliers, industries, damage, all_cogs, bneck_idx

def animate_mc_snapshots(
    G,
    cogs_map,
    milestones=range(500, 10001, 500),
    pause_ms=10_000,
    p_fail=P_FAIL_BASE,
    cogs_per_vehicle=COGS_PER_VEHICLE_USD,
    root_node="Tesla",
    precomp=None,               # (suppliers, industries, damage, all_cogs, bneck_idx)
    baseline_units=1_773_443,
):
    # --- precompute (or reuse) ---
    if precomp is None:
        suppliers, industries, damage, all_cogs, bneck_idx = precompute_runs(
            G, cogs_map, total_runs=10_000, p_fail=p_fail, root=root_node, seed=1
        )
    else:
        suppliers, industries, damage, all_cogs, bneck_idx = precomp

    # shared convergence derived once
    shortfall = TOTAL_COGS_REFERENCE - all_cogs
    run_avg_shortfall = np.cumsum(shortfall) / np.arange(1, len(all_cogs)+1)

    # --- layout like your network_graph.py ---
    pos = {}
    for i, s in enumerate(suppliers):
        y = (i + 0.5) * (len(industries) / max(len(suppliers), 1))
        pos[s] = (0.0, y)
    for i, ind in enumerate(industries):
        pos[ind] = (1.0, i + 0.5)
    pos[root_node] = (2.0, len(industries) / 2.0)

    # --- figure ---
    fig, (ax_net, ax_conv) = plt.subplots(2, 1, figsize=(16, 10),
                                          gridspec_kw={"height_ratios": [3, 1]})
    ax_net.set_title("Tesla Supply Chain Network", fontsize=18)

    # nodes & labels (bold, as you asked earlier)
    ax_net.scatter([pos[s][0] for s in suppliers], [pos[s][1] for s in suppliers], s=60,  c="#87CEFA")
    ax_net.scatter([pos[i][0] for i in industries], [pos[i][1] for i in industries], s=900, c="#90EE90")
    ax_net.scatter([pos[root_node][0]], [pos[root_node][1]], s=2000, c="#FFD700")
    for n in suppliers + industries + [root_node]:
        ax_net.text(pos[n][0], pos[n][1], str(n), fontsize=8, fontweight='bold', ha="center", va="center")

    # static industry->Tesla edges
    for ind in industries:
        ax_net.plot([pos[ind][0], pos[root_node][0]], [pos[ind][1], pos[root_node][1]],
                    color="lightgray", lw=1.0, zorder=1)

    # supplier->industry edges (we recolor with the synced run data)
    edge_lines = {}
    for ind in industries:
        for sup, _, _ in G.in_edges(ind, data=True):
            if sup in suppliers:
                ln, = ax_net.plot([pos[sup][0], pos[ind][0]], [pos[sup][1], pos[ind][1]],
                                  color="gray", lw=0.8, alpha=0.7, zorder=1)
                edge_lines[(sup, ind)] = ln

    leg_handles = [
        mpatches.Patch(color="#87CEFA", label="Suppliers (Layer 2)"),
        mpatches.Patch(color="#90EE90", label="Industries (Layer 1)"),
        mpatches.Patch(color="#FFD700", label="Tesla (Layer 0)"),
    ]
    ax_net.legend(handles=leg_handles, loc="center left", bbox_to_anchor=(0.92, 0.12),
                  frameon=True, shadow=True, fontsize=10)
    ax_net.set_axis_off()

    # info box (top-right)
    summary_txt = ax_net.text(
        0.78, 0.95, "", transform=ax_net.transAxes, va="top", ha="left",
        fontsize=11, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.95)
    )
    severity_texts = []

    # --- bottom convergence (full curve + status box) ---
    ax_conv.set_title("Monte Carlo Convergence (Running Average Shortfall)", fontsize=14)
    ax_conv.set_xlabel("Number of Simulations")
    ax_conv.set_ylabel("Average Shortfall ($ Billions)")
    ax_conv.grid(True, alpha=0.3)

    x = np.arange(1, len(all_cogs)+1)
    y = run_avg_shortfall / 1e9
    ax_conv.plot(x, y, lw=1.5)
    ax_conv.set_xlim(1, len(all_cogs))
    ypad = (y.max() - y.min()) * 0.1 + 1e-6
    ax_conv.set_ylim(y.min() - ypad, y.max() + ypad)
    marker, = ax_conv.plot([], [], "o", ms=8)

    conv_box = ax_conv.text(0.02, 0.96, "", transform=ax_conv.transAxes,
                            va="top", ha="left",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                            fontsize=10)

    # helper: nice percent without “-0%”
    def fmt_0pct(x, eps=5e-6):
        return f"{(0.0 if abs(x) < eps else x):.0%}"

    frames = list(milestones)
    sup_idx = {s:i for i, s in enumerate(suppliers)}

    def _update(milestone):
        nonlocal severity_texts
        # clear severity labels
        for t in severity_texts:
            t.remove()
        severity_texts = []

        idx = min(milestone - 1, len(all_cogs) - 1)

        # bottom: move marker + status
        marker.set_data([milestone], [y[idx]])
        curr_avg_shortfall = run_avg_shortfall[idx] / 1e9  # billions
        curr_bneck = industries[int(bneck_idx[idx])]

        # top: recolor supplier->industry edges from the SAME run
        row = damage[idx]
        for ln in edge_lines.values():
            ln.set_color("gray"); ln.set_linewidth(0.8); ln.set_alpha(0.7)

        for ind in industries:
            for sup, _, _ in G.in_edges(ind, data=True):
                if sup not in sup_idx:
                    continue
                sev = float(row[sup_idx[sup]])
                if sev > 0:
                    ln = edge_lines.get((sup, ind))
                    if ln is None:
                        continue
                    ln.set_color("red"); ln.set_linewidth(2.0); ln.set_alpha(0.95)
                    xm = (pos[sup][0] + pos[ind][0]) / 2.0
                    ym = (pos[sup][1] + pos[ind][1]) / 2.0
                    severity_texts.append(
                        ax_net.text(xm, ym, f"{sev*100:.0f}%", fontsize=8, color="red",
                                    ha="center", va="bottom")
                    )

        # summary box (numbers from the SAME run)
        tot   = float(all_cogs[idx])
        units = (tot / cogs_per_vehicle) if cogs_per_vehicle > 0 else 0.0
        impact = (baseline_units - units) / baseline_units
        impact_str = fmt_0pct(impact)
        bneck = industries[int(bneck_idx[idx])]
        summary_txt.set_text(
            f"Run: {milestone:,}\n"
            f"Total COGS: ${tot/1e9:.2f}B\n"
            f"No. of Units produced: {units:,.0f}\n"
            f"Impact on units: {impact_str}\n"
            f"Avg shortfall to {milestone:,}: ${curr_avg_shortfall:.2f}B\n"
            f"Bottleneck: {bneck}"
        )

        # stop cleanly at the end
        if milestone == frames[-1]:
            ani.event_source.stop()

        return [marker, summary_txt, conv_box] + list(edge_lines.values()) + severity_texts

    ani = FuncAnimation(fig, _update, frames=frames, interval=pause_ms,
                        blit=False, repeat=False, cache_frame_data=False)
    return ani, fig


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


    #before animation
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


