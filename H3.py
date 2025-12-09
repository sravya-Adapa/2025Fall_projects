from typing import Dict,Tuple,List,Optional,Any
import math
from Simulation import *

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
1. Loads the pickled NetworkX graph and asserts it’s directed, preventing accidental use of an undirected graph.

2. Keeps the object intact (no mutation/conversion), so downstream code can rely on directionality safely.
"""

def load_graph_from_pickle(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_digraph(obj: Any) -> nx.DiGraph:
    """Accept a graph or a path; load if needed and assert directedness."""

    G = load_graph_from_pickle(obj) if isinstance(obj, str) else obj
    if not nx.is_directed(G):
        raise TypeError("Graph is not directed; expected nx.DiGraph.")
    return G  # type: ignore[return-value]


"""
Goals achieved in below functions:
1. Pulls supplier to industry edge weights and normalizes them to shares in [0,1] (handles percent inputs too).

2. Encodes the layer-2 to layer-1 direction explicitly, so all later math uses the right flow.
"""

def _w01(x: float) -> float:
    """Coerce stored weights to [0,1] even if your file stores percent (e.g., 23.5 -> 0.235)."""
    return x / 100.0 if x > 1.0 else x

def get_industry_suppliers_and_shares(G: nx.DiGraph, industry: str) -> Dict[str, float]:
    """
    Return {supplier: share_in_[0,1]} for incoming edges to an industry.
    Uses direction: suppliers → industry.
    """
    if not nx.is_directed(G):
        raise TypeError("Expected directed graph in get_industry_suppliers_and_shares.")
    pairs: List[Tuple[str, float]] = []
    for sup, _, d in G.in_edges(industry, data=True):
        w = _w01(float(d.get("weight", 0.0)))
        if w > 0.0:
            pairs.append((sup, w))
    if not pairs:
        return {}
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return {}
    return {sup: w / tot for sup, w in pairs}

"""
Goals achieved in below functions:
1. Builds independent per-supplier severity series (fail Bernoulli, severities on failures) shared by both 
arms for a paired test.

2. Creates the common-event process (event flags + additive shock magnitudes) used only in the clustered arm.
"""

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
      - Bernoulli failure flags (unused here but kept for parity)
      - severity array s_i[r] in [0,1] where non-failures have 0, failures ~ U[low, high]
    Returned as: {supplier: (fails_bool, severities_float)}
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

def build_common_event(num_runs: int, p_event: float, shock_low: float, shock_high: float, seed: int = 999):
    """
    Common-shock process for the clustered arm:
      - event_flags[r] ~ Bernoulli(p_event)
      - shock[r] ~ U[shock_low, shock_high] when event occurs else 0
    """
    rng = np.random.default_rng(seed)
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
    seed_draws: int = 123,
    seed_event: int = 999,
) -> Dict[str, Any]:
    """
    Compare independent vs clustered failures using the same independent draws (paired design).
    A common event adds an extra severity S to all suppliers feeding `cluster_industry` on that run.
    """

    # Precompute structures & supplier cluster (the set feeding the target industry)
    industries, orig_cogs, ind_sup = _prepare_industry_inputs(G, cogs_map)
    shares = get_industry_suppliers_and_shares(G, cluster_industry)
    if not shares:
        raise ValueError(f"No suppliers found for industry '{cluster_industry}'.")
    cluster_suppliers = list(shares.keys())

    # Shared random draws for both arms (pairing)
    draws = build_independent_draws(G, num_runs=num_runs, p_fail=p_fail, seed=seed_draws)

    # Common-shock process for the clustered arm
    event_flags, event_shock = build_common_event(
        num_runs=num_runs, p_event=p_event,
        shock_low=shock_low, shock_high=shock_high, seed=seed_event
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
        import scipy.stats as st
        p_value_mean = 2 * st.t.sf(abs(t_stat), df=num_runs - 1)
    except Exception:
        from math import erf, sqrt
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
    """Overlay histograms and QQ plot of paired differences (clustered - independent)."""
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
        from scipy.stats import norm
        theo = norm.ppf(q, loc=mu, scale=sigma)
    except Exception:
        from statistics import NormalDist
        nd = NormalDist(mu, sigma if sigma > 0 else 1.0)
        theo = np.array([nd.inv_cdf(float(p)) for p in q])

    plt.subplot(1, 2, 2)
    plt.plot(theo, emp, "o", markersize=3)
    lo = min(theo.min(), emp.min())
    hi = max(theo.max(), emp.max())
    plt.plot([lo, hi], [lo, hi], lw=1.2)  # 45°
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Empirical quantiles")
    plt.title(f"QQ Plot of Paired Differences {title_suffix}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "supply_chain_graph", "tesla_supply_chain_graph_2024.pkl")

    # Load directed graph + COGS
    G = ensure_digraph(graph_path)
    industry_cogs = load_industry_cogs()

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
        seed_draws=123, # fixes the base (independent) failure/severity RNG for reproducibility.
        seed_event=999, # fixes the event timing and shock-size RNG, kept separate so you don’t entangle randomness streams
    )

    # Report (concise; mirrors H2 style)
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



