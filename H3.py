from helper_functions import *

"""
Hypothesis 3: When several suppliers fail together due to a common event (same region or shared sub-tier), 
the simulated production loss will be much higher than under independent failures, with p95 loss rising sharply.

There is one regional shock hits the Computer and electronics products industry, so due to which multiple upstream suppliers 
to that industry are disrupted at once. In simulations, when this event occurs (with probability 𝑝event), the whole cluster 
fails together with drawn severities, while all other suppliers across the network still fail independently as in the baseline.
Hypothesis 3 predicts that this correlated cluster shock will push both the mean loss and especially the p95 loss 
markedly higher than the independent-failure baseline.
"""

"""
Goals achieved in below functions:
1. Builds independent per-supplier severity series (fail Bernoulli, severities on failures) shared by both 
arms for a paired test.

2. Creates the common-event process (event flags + additive shock magnitudes) used only in the clustered arm.
"""

def build_common_event(num_runs: int, p_event: float, shock_low: float, shock_high: float):
    """
    Common-shock process for the clustered arm:
      - event_flags[r] ~ Bernoulli(p_event)
      - shock[r] ~ U[shock_low, shock_high] when event occurs else 0

    :param num_runs: Number of Monte Carlo runs.
    :param p_event: Probability the common event occurs on a given run (0–1).
    :param shock_low: Lower bound for the additive shock severity (inclusive).
    :param shock_high: Upper bound for the additive shock severity (inclusive).
    :return: Tuple (event_flags: np.ndarray[bool], event_shock: np.ndarray[float]).

    >>> flags, shock = build_common_event(num_runs=5, p_event=0.5,shock_low=0.3, shock_high=0.6)
    >>> len(flags)
    5
    >>> len(shock)
    5
    >>> all(shock[~flags] == 0)
    True
    >>> all((shock[flags] >= 0.3) & (shock[flags] <= 0.6))
    True
    """
    rng = np.random.default_rng()
    flags = rng.random(num_runs) < p_event
    shock = np.zeros(num_runs, dtype=np.float32)
    if flags.any():
        shock[flags] = rng.uniform(shock_low, shock_high, size=flags.sum()).astype(np.float32)
    return flags, shock


"""
Goals achieved in below functions:
1. Precomputes industries list, baseline COGS, and supplier-share lists to make the simulation inner loop lean.

2. Runs two simulations with the same base draws: independent (control) vs clustered (adds shock to target-industry 
suppliers, clamped to 1).
"""

def _prepare_industry_inputs(
    G: nx.DiGraph,
    cogs_map: Dict[str, float],
    root: str = "Tesla",
) -> Tuple[List[str], Dict[str, float], Dict[str, List[Tuple[str, float]]]]:
    """
    Prepare:
      industries: list of layer-1 industries that feed Tesla
      orig_cogs:  industry -> baseline $ value
      ind_sup:    industry -> list of (supplier, share)

    :param G: Directed supply-chain graph (supplier → industry → root).
    :param cogs_map: Mapping from industry name to baseline COGS dollars.
    :param root: Name of the root node (default "Tesla").

    :return: (industries, orig_cogs, ind_sup) as described above.
    :raises: KeyError if an industry in the graph is missing from cogs_map.
    :notes: Supplier-edge weights may be stored as percents; they are coerced to [0,1] and renormalized.

    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> G.add_edge("S1", "I1", weight=0.4)
    >>> G.add_edge("S2", "I1", weight=0.6)
    >>> G.add_edge("I1", "Tesla")
    >>> G.add_edge("S3", "I2", weight=1.0)
    >>> G.add_edge("I2", "Tesla")
    >>> cogs_map = {"I1": 10.0, "I2": 5.0}
    >>> industries, orig_cogs, ind_sup = _prepare_industry_inputs(G, cogs_map, root="Tesla")


    >>> sorted(industries)
    ['I1', 'I2']

    >>> orig_cogs == {'I1': 10.0, 'I2': 5.0}
    True

    >>> sorted(ind_sup["I1"])
    [('S1', 0.4), ('S2', 0.6)]

    >>> sorted(ind_sup["I2"])
    [('S3', 1.0)]
    """
    industries = [n for n in G.predecessors(root)]
    orig_cogs = {ind: float(cogs_map[ind]) for ind in industries}
    ind_sup: Dict[str, List[Tuple[str, float]]] = {}
    for ind in industries:
        shares = get_industry_suppliers_and_shares(G, ind)
        ind_sup[ind] = sorted(shares.items())
    return industries, orig_cogs, ind_sup

def simulate_independent_vs_clustered(
    industries: List[str],
    orig_cogs: Dict[str, float],
    ind_sup: Dict[str, List[Tuple[str, float]]],
    draws: Dict[str, Tuple[np.ndarray, np.ndarray]],
    num_runs: int,
    cluster_industry: str,
    cluster_suppliers: List[str],
    event_flags: np.ndarray,
    event_shock: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute total COGS arrays for:
      - independent arm (ctrl_out)
      - clustered arm  (clust_out) where, on event runs, suppliers in cluster_industry get +shock (clamped at 1).

    NOTE: The extra shock is applied only to supplier contributions into `cluster_industry` as discussed.

    :param industries: List of layer-1 industries feeding the root.
    :param orig_cogs: Mapping industry -> baseline COGS dollars.
    :param ind_sup: Mapping industry -> list of (supplier, share01).
    :param draws: Mapping supplier -> (fail_flags, base_severity) arrays for all runs.
    :param num_runs: Number of Monte Carlo runs.
    :param cluster_industry: Industry where the event shock applies.
    :param cluster_suppliers: Subset of suppliers (by name) feeding `cluster_industry` to receive the extra shock.
    :param event_flags: Boolean array indicating runs when the common event occurs.
    :param event_shock: Float array of additive shock magnitudes per run (0 when no event).

    :return: (ctrl_out, clust_out), two np.ndarray of length num_runs with total COGS per run.
    :notes: The two arms share the same `draws` (paired design) to isolate the effect of clustering.
    """
    ctrl_out  = np.zeros(num_runs, dtype=np.float64)
    clust_out = np.zeros(num_runs, dtype=np.float64)
    cluster_set = set(cluster_suppliers)

    for r in range(num_runs):
        total_ctrl = 0.0
        total_clst = 0.0
        extra = float(event_shock[r]) if event_flags[r] else 0.0

        for ind in industries:
            pct_ctrl = 1.0
            pct_clst = 1.0

            for sup, share in ind_sup[ind]:
                _, sev = draws.get(sup, (None, None))
                s = float(sev[r]) if sev is not None else 0.0

                # Independent arm (no extra common shock)
                pct_ctrl -= share * s

                # Clustered arm: only for contributions into the targeted industry
                if ind == cluster_industry and sup in cluster_set and extra > 0.0:
                    s_eff = min(1.0, s + extra)
                else:
                    s_eff = s
                pct_clst -= share * s_eff

            if pct_ctrl < 0.0: pct_ctrl = 0.0
            if pct_clst < 0.0: pct_clst = 0.0

            total_ctrl += orig_cogs[ind] * pct_ctrl
            total_clst += orig_cogs[ind] * pct_clst

        ctrl_out[r]  = total_ctrl
        clust_out[r] = total_clst

    return ctrl_out, clust_out

"""
Goals achieved in below functions:
1. Orchestrates the full experiment: builds draws/events, runs both arms, converts to loss vs baseline.

2. Computes mean loss increase (paired t-test) and p95 tail increase (paired bootstrap) and returns a clean 
result dict + verdict.
"""

def run_h3_experiment(
    G: nx.DiGraph,
    cogs_map: Dict[str, float],
    cluster_industry: str = "Computer and electronic products",
    num_runs: int = NUM_SIMS,
    p_fail: float = P_FAIL_BASE,
    # Common-event controls:
    p_event: float = 0.10,          # probability the cluster event fires on a run
    shock_low: float = 0.30,        # additive shock S ~ U[low, high] when event occurs
    shock_high: float = 0.60,
) -> Dict[str, Any]:
    """
    Compare independent vs clustered failures using the same independent draws (paired design).
    A common event adds an extra severity S to all suppliers feeding `cluster_industry` on that run.

    :param G: Directed supply-chain graph (supplier → industry → Tesla).
    :param cogs_map: Mapping industry -> baseline COGS dollars.
    :param cluster_industry: Industry whose suppliers receive the common additive shock.
    :param num_runs: Number of Monte Carlo runs.
    :param p_fail: Independent per-supplier failure probability in both arms.
    :param p_event: Probability the common event occurs on a run (clustered arm only).
    :param shock_low: Lower bound of additive shock (inclusive) when the event occurs.
    :param shock_high: Upper bound of additive shock (inclusive) when the event occurs.


    :return: Dict with keys such as:
             {
               "indep_loss", "clustered_loss",
               "mean_diff", "t_stat", "p_value_mean",
               "indep_p95", "clustered_p95", "p95_diff", "p_value_tail",
               "p_event", "shock_range", "cluster_industry", "num_runs"
             }
    """

    # Precompute structures & supplier cluster (the set feeding the target industry)
    industries, orig_cogs, ind_sup = _prepare_industry_inputs(G, cogs_map)
    shares = get_industry_suppliers_and_shares(G, cluster_industry)
    if not shares:
        raise ValueError(f"No suppliers found for industry '{cluster_industry}'.")
    cluster_suppliers = list(shares.keys())

    # Shared random draws for both arms (pairing)
    draws = build_independent_draws(G, num_runs=num_runs, p_fail=p_fail)

    # Common-shock process for the clustered arm
    event_flags, event_shock = build_common_event(
        num_runs=num_runs, p_event=p_event,
        shock_low=shock_low, shock_high=shock_high
    )

    # Simulate both arms
    ctrl_cogs, clst_cogs = simulate_independent_vs_clustered(
        industries, orig_cogs, ind_sup, draws, num_runs,
        cluster_industry=cluster_industry,
        cluster_suppliers=cluster_suppliers,
        event_flags=event_flags, event_shock=event_shock
    )

    # Convert to losses vs baseline
    ctrl_loss = TOTAL_COGS_REFERENCE - ctrl_cogs
    clst_loss = TOTAL_COGS_REFERENCE - clst_cogs

    # Mean difference & CI (paired)
    def _mean_ci(x: np.ndarray, level: float = 0.95) -> Tuple[float, float, float]:
        """
        Compute the sample mean and a two-sided (1−α) confidence interval via
        a normal (z) approximation.

        :param x: 1D array of observations.
        :param level: Desired confidence level in (0,1). Currently only 0.95 is supported by the fixed z constant,
        to support other levels, replace the hard-coded z with the appropriate quantile.
        :return: Tuple (mean, lower_bound, upper_bound).
        """
        m = float(x.mean())
        s = float(x.std(ddof=1))
        z = 1.959963984540054  # 97.5% quantile of N(0,1)
        half = z * s / math.sqrt(len(x))
        return m, m - half, m + half

    ctrl_mean, ctrl_lo, ctrl_hi = _mean_ci(ctrl_loss)
    clst_mean, clst_lo, clst_hi = _mean_ci(clst_loss)

    diff = clst_loss - ctrl_loss  # >0 means clustered is worse (as hypothesis predicts)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    t_stat = mean_diff / (sd_diff / math.sqrt(num_runs)) if sd_diff > 0 else float("inf")
    # Try scipy t; fall back to normal approx
    try:
        p_value_mean = 2 * stats.t.sf(abs(t_stat), df=num_runs - 1)
    except Exception:
        p_value_mean = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))

    # Tail (p95) increase & paired bootstrap p-value
    ctrl_p95 = float(np.percentile(ctrl_loss, 95))
    clst_p95 = float(np.percentile(clst_loss, 95))
    p95_increase = clst_p95 - ctrl_p95  # >0 supports H3

    rng = np.random.default_rng(2024)
    B = 1000
    idx = np.arange(num_runs)
    diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        resample = rng.choice(idx, size=num_runs, replace=True)
        diffs[b] = (np.percentile(clst_loss[resample], 95) -
                    np.percentile(ctrl_loss[resample], 95))
    # two-sided p-value for H0: no tail increase
    p_value_tail = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())

    verdict = "SUPPORTS" if (p95_increase > 0 and p_value_tail < 0.05) else "DOES NOT SUPPORT"

    return {
        "industry": cluster_industry,
        "num_runs": num_runs,
        "p_fail": p_fail,
        "p_event": p_event,
        "shock_low": shock_low,
        "shock_high": shock_high,
        "cluster_suppliers": cluster_suppliers,
        "control_loss": ctrl_loss,
        "clustered_loss": clst_loss,
        "ctrl_mean_CI": (ctrl_lo, ctrl_hi),
        "clst_mean_CI": (clst_lo, clst_hi),
        "mean_increase": mean_diff,
        "t_stat": t_stat,
        "p_value_mean": p_value_mean,
        "ctrl_p95": ctrl_p95,
        "clst_p95": clst_p95,
        "p95_increase": p95_increase,
        "p_value_tail": p_value_tail,
        "verdict": verdict,
    }


"""
Goals achieved in below functions:
1. Plots overlaid loss histograms (independent vs clustered) to show magnitude and spread differences.

2. Adds a QQ plot of paired differences to sanity-check normality/shape assumptions behind the t-test.
"""
def quick_plots(loss_indep: np.ndarray, loss_clustered: np.ndarray, title_suffix: str = ""):
    """Overlay histograms and QQ plot of paired differences (clustered - independent).

    :param loss_indep: Array of per-run losses under the independent arm.
    :param loss_clustered: Array of per-run losses under the clustered arm.
    :param title_suffix: Optional string appended to figure titles.

    :return: None (plots to the active Matplotlib backend).
    """
    plt.figure(figsize=(11, 4))

    # 1) Loss histograms
    plt.subplot(1, 2, 1)
    bins = 40
    plt.hist(loss_indep / 1e9, bins=bins, alpha=0.6, label="Independent loss (B$)")
    plt.hist(loss_clustered / 1e9, bins=bins, alpha=0.6, label="Clustered loss (B$)")
    plt.xlabel("Loss (Billions USD)")
    plt.ylabel("Frequency")
    plt.title(f"Loss Distributions {title_suffix}")
    plt.legend()

    # 2) QQ of paired differences vs Normal(μ,σ)
    diff = loss_clustered - loss_indep
    mu, sigma = diff.mean(), diff.std(ddof=1)
    q = np.linspace(0.01, 0.99, 99)
    emp = np.quantile(diff, q)
    try:
        theo = norm.ppf(q, loc=mu, scale=sigma)
    except Exception:
        nd = NormalDist(mu, sigma if sigma > 0 else 1.0)
        theo = np.array([nd.inv_cdf(float(p)) for p in q])

    plt.subplot(1, 2, 2)
    plt.plot(theo, emp, marker="o", linestyle="none", markersize=3,
             label="Paired diffs (empirical)")
    lo = min(theo.min(), emp.min());
    hi = max(theo.max(), emp.max())
    plt.plot([lo, hi], [lo, hi], lw=1.2, label="45° normal reference") # 45° line
    plt.xlabel("Theoretical quantiles");
    plt.ylabel("Empirical quantiles")
    plt.title(f"QQ Plot of Paired Differences {title_suffix}")
    plt.legend(loc="lower right", frameon=True, fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "supply_chain_graph", "tesla_supply_chain_graph_2024.pkl")

    # Load directed graph + COGS
    G = load_graph_from_pickle(graph_path)
    industry_cogs = load_cogs_per_industry()

    # Run H3 on the chosen cluster industry
    res = run_h3_experiment(
        G,
        industry_cogs,
        cluster_industry="Computer and electronic products",  # your request
        num_runs=NUM_SIMS,
        p_fail=P_FAIL_BASE,
        # Event controls (tune these if you want stronger/weaker correlation)
        p_event=0.10, # Here 10% or 1000 runs experience a cluster-wide shock for the chosen industry
        shock_low=0.30, # when that shock happens, each supplier in the cluster gets an extra severity this is the lowest value
        shock_high=0.60, # when that shock happens, each supplier in the cluster gets an extra severity this is the highest value
    )

    # Report
    print("\n=== Common-Shock (H3) Experiment Report ===")
    print(f"Industry (cluster): {res['industry']}")
    print(f"Runs: {res['num_runs']:,} | p_fail: {res['p_fail']:.3f} | p_event: {res['p_event']:.2f} "
          f"| shock ~ U[{res['shock_low']:.2f},{res['shock_high']:.2f}]")
    print(f"Cluster suppliers (count): {len(res['cluster_suppliers'])}")

    lo_c, hi_c = res["ctrl_mean_CI"]
    lo_t, hi_t = res["clst_mean_CI"]
    print(f"Mean loss (indep) 95% CI: (${lo_c/1e9:.3f}B, ${hi_c/1e9:.3f}B)")
    print(f"Mean loss (cluster) 95% CI: (${lo_t/1e9:.3f}B, ${hi_t/1e9:.3f}B)")
    print(f"Mean increase (cluster - indep): ${res['mean_increase']/1e9:.3f}B")
    print(f"Paired t-statistic: {res['t_stat']:.3f},  p-value: {res['p_value_mean']:.3g}")

    print(f"P95 loss (indep):    ${res['ctrl_p95']/1e9:.3f}B")
    print(f"P95 loss (cluster):  ${res['clst_p95']/1e9:.3f}B")
    print(f"P95 increase:        ${res['p95_increase']/1e9:.3f}B")
    print(f"Tail p-value (paired bootstrap): {res['p_value_tail']:.3g}")

    print(f"\nVerdict on H3: {res['verdict']}\n")

    # Quick visuals
    quick_plots(res["control_loss"], res["clustered_loss"], title_suffix="(H3 common shock)")


"""
Disclaimer:
This project was developed with assistance from GPT. 
GPT was used to:
  • suggest statistical methods for hypothesis-testing,
  • help to debug errors and improve code structure,
  • refine syntax for Python, NumPy, NetworkX, and plotting libraries,
  • helped with complex doctests,
  • and clarify conceptual questions during development.
"""
