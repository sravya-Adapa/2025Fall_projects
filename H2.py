from helper_functions import *




"""
Hypothesis 2: Within a industry category, spreading volume across more suppliers (lower concentration/HHI) 
will reduce both average loss and tail risk compared with keeping one dominant supplier at the same total 
category weight.
"""

"""
Goals achieved in below functions:
1. Reads supplier to industry edge weights and normalizes them to shares in [0,1].
2. Computes HHI, HHI (Herfindahl–Hirschman Index) which measures concentration, sum of the squared shares.
Here: for one industry, take each supplier’s share of that industry (fractions that sum to 1) and compute 
HHI = ∑𝑖 𝑠𝑖^2.Lower HHI → diversified (e.g., equal split); higher HHI → concentrated (one dominant supplier).
3. Defined treatment weights: perfectly uniform on same suppliers.
"""

# ========== Edge-Weight utilities ==========

def hhi(shares01: Dict[str, float]) -> float:
    """Herfindahl–Hirschman Index on fractional shares; HHI = ∑ s_i^2.

    >>> hhi({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    0.25
    >>> round(hhi({"A": -0.2, "B": 0.3, "C": 0.5}), 6)
    0.38
    >>> hhi({})
    0
    """
    return sum(s * s for s in shares01.values())


# ========== Reweighting for treatment and control categories ==========

def reweight_suppliers_uniform(shares01: Dict[str, float]) -> Dict[str, float]:
    """
    Treatment: keep the same supplier set, assign equal shares (min HHI for fixed set).
    >>> reweight_suppliers_uniform({})
    {}

    >>> reweight_suppliers_uniform({"A": 0.3})
    {'A': 1.0}

    >>> reweight_suppliers_uniform({"A": 0.7, "B": 0.3})
    {'A': 0.5, 'B': 0.5}

    >>> out = reweight_suppliers_uniform({"X": 0.2, "Y": 0.5, "Z": 0.3})
    >>> sorted((k, round(v, 6)) for k, v in out.items())
    [('X', 0.333333), ('Y', 0.333333), ('Z', 0.333333)]
    """
    if not shares01:
        return {}
    n = len(shares01)
    eq = 1.0 / n
    return {k: eq for k in shares01}

def reweight_suppliers_concentrated(
    shares01: Dict[str, float],
    top_frac: float = 0.80,
    pick: str = "largest",
    seed: int = 7,
) -> Dict[str, float]:
    """
    CONTROL: concentrated split inside the SAME supplier set.
    Default: 80% to the largest-current-share supplier; remaining 20% split randomly among the others.

    Args
    ----
    shares01 : dict of {supplier: share} that sum to 1 (baseline from graph)
    top_frac : share for the 'dominant' supplier (e.g., 0.80)
    pick     : "largest" or a supplier name (to force a specific supplier)
    seed     : RNG seed for reproducible splits among the rest

    >>> reweight_suppliers_concentrated({})
    {}

    >>> reweight_suppliers_concentrated({"A": 0.3})
    {'A': 1.0}

    >>> out = reweight_suppliers_concentrated({"A": 0.6, "B": 0.4}, top_frac=0.8, seed=1)
    >>> round(out["A"], 6)
    0.8
    >>> round(out["B"], 6)
    0.2
    >>> out = reweight_suppliers_concentrated({"A": 0.2, "B": 0.8}, pick="A", top_frac=0.7, seed=1)
    >>> round(out["A"], 6)
    0.7
    """
    if not shares01:
        return {}
    if len(shares01) == 1:
        # Only one supplier; it's 100% by definition.
        k = next(iter(shares01.keys()))
        return {k: 1.0}

    # Choose the dominant supplier
    if pick == "largest":
        top_supplier = max(shares01.items(), key=lambda kv: kv[1])[0]
    else:
        if pick not in shares01:
            raise ValueError(f"'pick' supplier '{pick}' not found in shares.")
        top_supplier = pick

    others = [s for s in shares01.keys() if s != top_supplier]
    rng = np.random.default_rng(seed)
    # Random split for the remaining (1 - top_frac) using Dirichlet
    if len(others) == 1:
        other_alloc = np.array([1.0])
    else:
        other_alloc = rng.dirichlet(np.ones(len(others)))

    ctrl = {top_supplier: float(top_frac)}
    remainder = 1.0 - float(top_frac)
    for s, a in zip(others, other_alloc):
        ctrl[s] = float(remainder * a)

    # Numerical hygiene: renormalize to exactly 1
    total = sum(ctrl.values())
    ctrl = {k: v / total for k, v in ctrl.items()}
    return ctrl

"""
Goals achieved in below functions:
1. The function "build_common_draws" makes one randomness table and uses it for both control & treatment category.

2. The function "simulate_with_draws" is a tight loop that reads those severities and the shares to compute total COGS per run.
"""

def _prepare_industry_inputs(
    G: nx.DiGraph,
    cogs_map: Dict[str, float],
    overrides_for_industry: Optional[Dict[str, Dict[str, float]]] = None,
    root: str = "Tesla",
) -> Tuple[List[str], Dict[str, float], Dict[str, List[Tuple[str, float]]]]:
    """
    Precompute per-industry structural inputs so we can simulate fast:
      - industries list
      - original COGS per industry
      - supplier share list per industry: {industry: [(supplier, share01), ...], ...}
    If overrides_for_industry is provided, it should be {industry: {supplier: share01,...}}.
    >>> import networkx as nx
    >>> G = nx.DiGraph()
    >>> # Tesla has two industries feeding it
    >>> G.add_edge("IndA", "Tesla")
    >>> G.add_edge("IndB", "Tesla")
    >>> # Suppliers feeding IndA
    >>> G.add_edge("S1", "IndA", weight=0.6)
    >>> G.add_edge("S2", "IndA", weight=0.4)
    >>> # Suppliers feeding IndB
    >>> G.add_edge("S3", "IndB", weight=1.0)
    >>> cogs = {"IndA": 100.0, "IndB": 50.0}

    >>> inds, orig_cogs, ind_sup = _prepare_industry_inputs(G, cogs)
    >>> sorted(inds)
    ['IndA', 'IndB']

    >>> orig_cogs == {"IndA": 100.0, "IndB": 50.0}
    True

    # Supplier shares should reflect normalized weights
    >>> ind_sup["IndA"]
    [('S1', 0.6), ('S2', 0.4)]

    >>> ind_sup["IndB"]
    [('S3', 1.0)]

    >>> overrides = {"IndB": {"S3": 5.0}}
    >>> _, _, ind_sup3 = _prepare_industry_inputs(G, cogs, overrides_for_industry=overrides)
    >>> ind_sup3["IndB"]
    [('S3', 1.0)]

    >>> _, _, ind_sup4 = _prepare_industry_inputs(G, cogs, overrides_for_industry={})
    >>> ind_sup4["IndA"]
    [('S1', 0.6), ('S2', 0.4)]
    """
    industries = [n for n in G.predecessors(root)]
    orig_cogs = {ind: float(cogs_map[ind]) for ind in industries}
    ind_sup_shares = {}
    for ind in industries:
        base = get_industry_suppliers_and_shares(G, ind)
        if overrides_for_industry and ind in overrides_for_industry:
            # keep only suppliers that exist in base; renormalize just in case
            ov = {s: overrides_for_industry[ind].get(s, base.get(s, 0.0)) for s in base}
            tot = sum(ov.values())
            if tot > 0:
                ov = {k: v/tot for k, v in ov.items()}
            base = ov
        ind_sup_shares[ind] = sorted(base.items())
    return industries, orig_cogs, ind_sup_shares

def simulate_with_draws(
    industries: List[str],
    orig_cogs: Dict[str, float],
    ind_sup_shares: Dict[str, List[Tuple[str, float]]],
    draws: Dict[str, Tuple[np.ndarray, np.ndarray]],
    num_runs: int,
) -> np.ndarray:
    """
    Vectorized-ish loop: for each run r, compute available COGS:
        percentage_left[ind] = 1 - sum_over_sup( share * severity[r] )
        total_cogs[r] = sum( orig_cogs[ind] * percentage_left[ind] )
    (If a supplier didn't fail at run r, severity[r]==0, so it contributes nothing.)
    """
    out = np.zeros(num_runs, dtype=np.float64)
    for r in range(num_runs):
        total = 0.0
        for ind in industries:
            pct_left = 1.0
            for sup, share in ind_sup_shares[ind]:
                _, sev = draws.get(sup, (None, None))
                # sev may be None if something odd; guard
                if sev is not None:
                    pct_left -= share * float(sev[r])
            if pct_left < 0.0:
                pct_left = 0.0
            total += orig_cogs[ind] * pct_left
        out[r] = total
    return out

"""
Goals achieved in below functions:
1. Leaves the graph unchanged, treatment only reweights the chosen industry in memory.

2. Uses the same random shocks for control and treatment (paired test = higher power).

3. Reports HHI for both arms, mean loss reduction with CI + p-value, tail reduction + p-value, and a final verdict.
"""

def run_h2_experiment(
    G: nx.DiGraph,
    cogs_map: Dict[str, float],
    target_industry: str = "Computer and electronic products",
    num_runs: int = NUM_SIMS,
    p_fail: float = P_FAIL_BASE,
    seed: int = 123,
) -> Dict[str, any]:
    """
    Reweights suppliers ONLY inside target_industry; everything else unchanged.
    CONTROL: concentrated (80/20 random across the same supplier set).
    TREATMENT: uniform across the same supplier set.
    Uses the SAME random shocks for both arms (paired test). Returns metrics and series.
    """
    # Base supplier set from the graph (used only to know which suppliers exist)
    base_shares = get_industry_suppliers_and_shares(G, target_industry)
    if not base_shares:
        raise ValueError(f"No suppliers found feeding industry: {target_industry}")

    # CONTROL (concentrated) and TREATMENT (uniform)
    ctrl_shares = reweight_suppliers_concentrated(base_shares, top_frac=0.80, pick="largest", seed=seed)
    trt_shares  = reweight_suppliers_uniform(base_shares)

    base_hhi  = hhi(ctrl_shares)   # report as "baseline HHI" (control arm)
    treat_hhi = hhi(trt_shares)

    ctrl_overrides = {target_industry: ctrl_shares}
    trt_overrides  = {target_industry: trt_shares}

    # Precompute structures for both
    industries_ctrl, orig_cogs_ctrl, ind_sup_ctrl = _prepare_industry_inputs(G, cogs_map, ctrl_overrides)
    industries_trt,  orig_cogs_trt,  ind_sup_trt  = _prepare_industry_inputs(G, cogs_map, trt_overrides)

    # Common random numbers
    draws = build_independent_draws(G, num_runs=num_runs, p_fail=p_fail, seed=seed)

    # Simulate
    ctrl_cogs = simulate_with_draws(industries_ctrl, orig_cogs_ctrl, ind_sup_ctrl, draws, num_runs)
    trt_cogs  = simulate_with_draws(industries_trt,  orig_cogs_trt,  ind_sup_trt,  draws, num_runs)

    # Losses
    ctrl_loss = TOTAL_COGS_REFERENCE - ctrl_cogs
    trt_loss  = TOTAL_COGS_REFERENCE - trt_cogs

    # Mean + CI
    def _mean_ci(x: np.ndarray, level: float = 0.95) -> Tuple[float, float, float]:
        m = float(x.mean())
        s = float(x.std(ddof=1))
        z = 1.959963984540054  # ~N(0,1) 97.5% quantile
        half = z * s / math.sqrt(len(x))
        return m, m - half, m + half

    ctrl_mean, ctrl_lo, ctrl_hi = _mean_ci(ctrl_loss)
    trt_mean,  trt_lo,  trt_hi  = _mean_ci(trt_loss)

    # Paired t-test on per-run loss difference
    diff = ctrl_loss - trt_loss  # >0 means treatment reduced loss
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    t_stat = mean_diff / (sd_diff / math.sqrt(num_runs)) if sd_diff > 0 else float("inf")

    try:
        p_value_mean = 2 * stats.t.sf(abs(t_stat), df=num_runs - 1)
    except Exception:
        p_value_mean = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))

    # Tail (95th percentile) + paired bootstrap p-value
    ctrl_p95 = float(np.percentile(ctrl_loss, 95))
    trt_p95  = float(np.percentile(trt_loss, 95))
    p95_diff = ctrl_p95 - trt_p95   # >0 → treatment reduces tail loss

    rng = np.random.default_rng(2024)
    B = 1000
    idx = np.arange(num_runs)
    diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        resample = rng.choice(idx, size=num_runs, replace=True)
        diffs[b] = (np.percentile(ctrl_loss[resample], 95) -
                    np.percentile(trt_loss[resample], 95))
    p_value_tail = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())

    # Verdict
    supports_mean = (mean_diff > 0) and (p_value_mean < 0.05)
    supports_tail = (p95_diff  > 0) and (p_value_tail < 0.05)
    verdict = "SUPPORTS" if (supports_mean and supports_tail) else "DOES NOT SUPPORT"

    return {
        "target_industry": target_industry,
        "base_hhi": base_hhi,           # control HHI (concentrated)
        "treat_hhi": treat_hhi,         # treatment HHI (uniform)
        "control_loss": ctrl_loss,
        "treatment_loss": trt_loss,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_value_mean": p_value_mean,
        "ctrl_mean_CI": (ctrl_lo, ctrl_hi),
        "trt_mean_CI": (trt_lo, trt_hi),
        "ctrl_p95": ctrl_p95,
        "trt_p95": trt_p95,
        "p95_diff": p95_diff,
        "p_value_tail": p_value_tail,
        "verdict": verdict,
        "num_runs": num_runs,
    }

"""
Goals achieved in below functions:
1. Quick visualization plots (histogram + QQ) plots for this hypothesis testing
"""

def quick_plots(loss_ctrl: np.ndarray, loss_trt: np.ndarray, title_suffix: str = ""):
    """
    Simple overlay histogram and QQ-plot of paired differences for a quick visual check.
    """

    # 1) histogram overlay
    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    bins = 40
    plt.hist(loss_ctrl/1e9, bins=bins, alpha=0.6, label="Control loss (B$)")
    plt.hist(loss_trt/1e9,  bins=bins, alpha=0.6, label="Treatment loss (B$)")
    plt.xlabel("Loss (Billions USD)"); plt.ylabel("Frequency")
    plt.title(f"Loss Distributions {title_suffix}")
    plt.legend()

    # 2) QQ-plot of paired differences vs Normal(μ,σ) for a sanity check
    diff = (loss_ctrl - loss_trt)
    mu, sigma = diff.mean(), diff.std(ddof=1)
    q = np.linspace(0.01, 0.99, 99)
    emp = np.quantile(diff, q)

    try:
        theo = norm.ppf(q, loc=mu, scale=sigma)
    except Exception:
        # normal approx fallback
        nd = NormalDist(mu, sigma if sigma>0 else 1.0)
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
    # Adjust path to your directed graph
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "supply_chain_graph", "tesla_supply_chain_graph_2024.pkl")


    # Load & verify directedness
    G = load_graph_from_pickle(graph_path)

    # Load cogs per industry
    industry_cogs = load_cogs_per_industry()

    # All industries (Layer 1) connected to Tesla
    industries = [n for n in G.predecessors("Tesla")]

    # Collect a summary row per industry
    rows = []

    for ind in industries:
        res = run_h2_experiment(
            G, industry_cogs,
            target_industry=ind,
            num_runs=NUM_SIMS,
            p_fail=P_FAIL_BASE,
            seed=123
        )


        print("\n=== H2 Experiment Report ===")
        print(f"Industry: {res['target_industry']}")
        print(f"HHI (baseline):  {res['base_hhi']:.4f}")
        print(f"HHI (treatment): {res['treat_hhi']:.4f}")
        print(f"Mean loss reduction (control - treatment): ${res['mean_diff'] / 1e9:.3f}B")
        print(f"Paired t-statistic: {res['t_stat']:.3f},  p-value: {res['p_value_mean']:.3g}")
        lo_c, hi_c = res["ctrl_mean_CI"]
        lo_t, hi_t = res["trt_mean_CI"]
        print(f"Control mean loss 95% CI:  (${lo_c / 1e9:.3f}B, ${hi_c / 1e9:.3f}B)")
        print(f"Treat.  mean loss 95% CI:  (${lo_t / 1e9:.3f}B, ${hi_t / 1e9:.3f}B)")
        print(f"P95 loss (control):  ${res['ctrl_p95'] / 1e9:.3f}B")
        print(f"P95 loss (treat.):   ${res['trt_p95'] / 1e9:.3f}B")
        print(f"Tail reduction (p95 ctrl - p95 trt): ${res['p95_diff'] / 1e9:.3f}B")
        print(f"Tail p-value (paired bootstrap): {res['p_value_tail']:.3g}")
        print(f"\nVerdict on hypothesis: {res['verdict']}\n")

        # same visuals, per industry
        quick_plots(
            res["control_loss"],
            res["treatment_loss"],
            title_suffix=f"(H2 test)"
        )

        # for table at the end, summary row
        rows.append({
            "Industry": res["target_industry"],
            "HHI_base": res["base_hhi"],
            "HHI_treat": res["treat_hhi"],
            "MeanLossReduction_B": res["mean_diff"] / 1e9,
            "t_stat": res["t_stat"],
            "p_mean": res["p_value_mean"],
            "P95_ctrl_B": res["ctrl_p95"] / 1e9,
            "P95_trt_B": res["trt_p95"] / 1e9,
            "TailReduction_B": res["p95_diff"] / 1e9,
            "p_tail": res["p_value_tail"],
            "Conclusion": ("TRUE" if res["verdict"] == "SUPPORTS" else "FALSE"),
            "Runs": res["num_runs"],
        })

    # Print a tidy table at the end
    if rows:
        tbl = pd.DataFrame(rows).sort_values(
            by=["Conclusion", "MeanLossReduction_B"], ascending=[False, False]
        )

        with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.expand_frame_repr", False,
                "display.float_format", lambda x: f"{x:,.3f}"
        ):
            print("\n=== HHI Experiment Summary Across Industries ===")
            print(tbl[
                      ["Industry", "HHI_base", "HHI_treat",
                       "MeanLossReduction_B", "t_stat", "p_mean",
                       "P95_ctrl_B", "P95_trt_B", "TailReduction_B", "p_tail",
                       "Conclusion", "Runs"]
                  ])



