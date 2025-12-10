from H1 import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import numpy as np

def precompute_runs(G, cogs_map, total_runs, p_fail, root, seed):
    """
    Precompute Monte Carlo draws and derived per-run aggregates for the live animation.

    • Identifies suppliers (in-degree==0) and industries (parents of `root`) from the directed graph.
    • For each run r ∈ {1..total_runs}, samples independent supplier failures with probability `p_fail`
      and assigns a severity in [0,1] (0 when not failed).
    • Propagates severities supplier→industry to compute remaining availability per industry and the
      total simulated COGS for that run.
    • Records a simple bottleneck index: the argmin industry by remaining availability in run r.

    :param G: nx.DiGraph Directed supply-chain graph (supplier → industry → `root`). Must be directed.
    :param cogs_map : Dict[str, float] Mapping from industry name to baseline COGS dollars.
    :param total_runs : int Number of Monte Carlo runs to precompute (e.g., 10_000).
    :param p_fail : float Per-supplier failure probability in each run (0–1).
    :param root : str Name of the root node (default "Tesla"); industries are G.predecessors(root).
    :param seed : int RNG seed to make precomputed draws reproducible.

    :return: Tuple[List[str], List[str], np.ndarray, np.ndarray, np.ndarray]
        (suppliers, industries, damage, all_cogs, bneck_idx) where:
          • suppliers : list[str]
          • industries: list[str]
          • damage    : float array of shape (total_runs, len(suppliers)),
                        per-run supplier severities in [0,1] (0 means no failure)
          • all_cogs  : float array of shape (total_runs,), total simulated COGS dollars per run
          • bneck_idx : int array of shape (total_runs,), index into `industries` of the most
                        constrained industry in each run
    """

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
    milestones=range(100, 10001, 100),
    pause_ms=10_000,
    p_fail=P_FAIL_BASE,
    cogs_per_vehicle=COGS_PER_VEHICLE_USD,
    root_node="Tesla",
    precomp=None,               # (suppliers, industries, damage, all_cogs, bneck_idx)
    baseline_units=1_773_443,
):
    """
    Two-panel animation for the Monte Carlo model: network snapshot (top) + convergence marker (bottom).

    • Top panel: Tesla supply network laid out in 3 columns (Suppliers → Industries → Tesla).
      Supplier→Industry edges that "fail" for the current frame are highlighted in red with a
      severity label. A summary box shows the milestone run number, total COGS, implied units,
      bottleneck industry, and impact-on-units.
    • Bottom panel: Running-average shortfall curve for all runs (drawn once) with a marker that
      advances to the current milestone (e.g., 100, 200, …, 10_000). This conveys convergence.

    :param G : nx.DiGraph Directed supply-chain graph (supplier → industry → root).
    :param cogs_map : Dict[str, float] Mapping from industry name to baseline COGS dollars.
    :param milestones : Iterable[int], The subset of run indices to show as frames (marker positions). Defaults to 100..10_000 step 100.
    :param pause_ms : int, optional Dwell time per frame in milliseconds (e.g., 10_000 = 10s per milestone).
    :param p_fail : float, optional Per-supplier failure probability used only for the *visual* red edge highlights when
    `precomp` is not driving the snapshot look. The convergence curve itself is precomputed.
    :param cogs_per_vehicle : float, optional COGS dollars per vehicle, used to compute implied units in the summary box.
    :param root_node : str, optional Name of the root node ("Tesla").
    :param precomp : Optional[Tuple], optional A tuple (suppliers, industries, damage, all_cogs, bneck_idx) returned by `precompute_runs`.
    If provided, the bottom convergence curve and summary stats use these exact precomputed values.
    :param baseline_units : int, optional Baseline production units, used to show "impact on units" versus baseline.

    :return: Tuple[matplotlib.animation.FuncAnimation, matplotlib.figure.Figure] (ani, fig) so the caller can keep a global
    reference to prevent GC and then call plt.show().
    """

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

    # --- Layout of the graph similar to initial proposed network graph ---

    pos = {}
    for i, s in enumerate(suppliers):
        y = (i + 0.5) * (len(industries) / max(len(suppliers), 1))
        pos[s] = (0.0, y)
    for i, ind in enumerate(industries):
        pos[ind] = (1.0, i + 0.5)
    pos[root_node] = (2.0, len(industries) / 2.0)


    # --- Figure ---
    fig, (ax_net, ax_conv) = plt.subplots(2, 1, figsize=(16, 10),
                                          gridspec_kw={"height_ratios": [3, 1]})
    ax_net.set_title("Tesla Supply Chain Network", fontsize=18)

    # nodes & labels
    ax_net.scatter([pos[s][0] for s in suppliers], [pos[s][1] for s in suppliers], s=60,  c="#87CEFA")
    ax_net.scatter([pos[i][0] for i in industries], [pos[i][1] for i in industries], s=900, c="#90EE90")
    ax_net.scatter([pos[root_node][0]], [pos[root_node][1]], s=2000, c="#FFD700")
    for n in suppliers + industries + [root_node]:
        ax_net.text(pos[n][0], pos[n][1], str(n), fontsize=8, fontweight='bold', ha="center", va="center")

    # static industry to Tesla root node edges design
    for ind in industries:
        ax_net.plot([pos[ind][0], pos[root_node][0]], [pos[ind][1], pos[root_node][1]],
                    color="lightgray", lw=1.0, zorder=1)

    # supplier to industry edges (need to recolor with the synced run data)
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

    # --- Bottom convergence (full curve + status box) layout ---
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

    # helper function to avoid “-0%” condition
    def fmt_0pct(x, eps=5e-6):
        """
        Format a percentage while suppressing the string '-0%'. Values whose absolute magnitude is < eps are
        treated as exactly 0.0 before formatting. This avoids artifacts from tiny negative rounding errors
        (e.g., -0.000001 → "0%").

        :param x : float Fraction to be rendered as a percentage (e.g., 0.034 → "3%").
        :param eps : float, Tolerance below which the value is coerced to 0.0. Default 5e-6.

        :return: str Percentage string with no '-0%' cases.
        """
        return f"{(0.0 if abs(x) < eps else x):.0%}"

    frames = list(milestones)
    sup_idx = {s:i for i, s in enumerate(suppliers)}

    def _update(milestone):
        """
            Render a single animation frame for the given milestone run index.

            Steps:
            1) Clear previous severity labels and reset supplier→industry edge styling.
            2) Move the convergence marker to `milestone` on the bottom plot.
            3) Sample (or read from `precomp`) per-supplier severities for the snapshot,
               color failing edges red and annotate with severity.
            4) Update the summary box (total COGS, implied units, bottleneck, impact-on-units).

            :param milestone: Iterable[int], The subset of run indices to show as frames (marker positions). Defaults to 100..10_000 step 100.

            :return: a list of Matplotlib artists to be redrawn by FuncAnimation.
        """
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

        # top: recolor supplier to industry edges from the SAME run and it is marked as red.
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

        # summary box layout for each run
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

    PRECOMP = precompute_runs(G, industry_cogs, total_runs=10_000, p_fail=P_FAIL_BASE, root='Tesla', seed=1)

    ANIM, FIG = animate_mc_snapshots(
        G, industry_cogs,
        milestones=range(100, 10001, 100),
        pause_ms=10_000,
        p_fail=P_FAIL_BASE,
        cogs_per_vehicle=COGS_PER_VEHICLE_USD,
        root_node="Tesla",
        precomp=PRECOMP,  # <<< sync!
        baseline_units=1_773_443,
    )
    plt.tight_layout()
    plt.show()
