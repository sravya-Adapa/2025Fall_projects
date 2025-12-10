# Monte Carlo Simulation of Supplier Disruptions and Their Impact on an EV Manufacturer’s Production

---

## Team Members
- **Dhyey Kasundra**  
- **Sravya Adapa**
  
---

## Project Type
**Type II – Original Monte Carlo Simulation**

---

## Table of Contents

- [Overview](#overview)
- [Objective](#objective)
- [Repository Structure](#repository-structure)
- [Data Preparation](#data-preparation)
- [Supply Chain Network Graph](#supply-chain-network-graph)
- [Monte Carlo Simulation Model](#monte-carlo-simulation-model)
- [Hypothesis 1](#hypothesis-1)
- [Hypothesis 2](#hypothesis-2)
- [Hypothesis 3](#hypothesis-3)
- [Limitations](#limitations)
- [Dependencies](#dependencies)
- [Data Sources and References](#data-sources-and-references)
  
---

## Overview
This project simulates how disruptions in a **giant electric vehicle manufacturing firm like Tesla**, in their supplier network, affect its ability to **produce vehicles and control production cost**.

An EV manufacturer depends on key component industries like **battery systems, power electronics, motors/drive units, body/metals, interior/trim, plastics/rubber, and electronics**. When supply shocks occur due to factory outages, quality recalls, logistics delays, or regional events, inputs from critical suppliers can drop suddenly. These disruptions constrain component availability at the plant and reduce finished-vehicle.

The model consists of a **two-layer network**, industry categories (batteries, electronics, metals, etc.) feeding the EV product, and named suppliers feeding each category. Category weights are derived from **BEA Input–Output (direct requirements)** as a public proxy for the bill of materials, and supplier names are mapped from public sources with clearly stated allocation assumptions. A **10,000 run Monte Carlo** injects random supplier failures to estimate **expected shortfall** in production cost and the number of units produced in 2024. Results are visualized with **NetworkX** and summarized using **95% confidence intervals, paired t-tests**, and **paired bootstrap p-values**.

---

## Objective
- Build a two-layer EV supply chain model using BEA category weights and named suppliers.
- Run Monte Carlo simulations for 10k runs to estimate production shortfall from supplier disruptions.
- Test hypothesis to answer questions:
    - Whether the top two categories contribute most towards the production loss?
    - Does spreading volume across more suppliers within an industry category reduce the production loss?
    - Do common-shock supplier failures create larger expected production loss than independent failures?
- Report clear evidence via summary tables and visual plots (network view, convergence, histograms, QQ plots) with statistical tests.

---

## Repository Structure
```
├── Tesla_specific_data/
│   ├── preprocessed_data/
│   │   └── component_weights_2024_percent.csv     # BEA shares scaled to 2024 COGS
│   ├── BEA_industry_to_commodity_updated_data.csv # BEA data
│   ├── Supplier_to_BEA_commodity_mapping.csv      # Named suppliers → BEA category map
│   ├── Tesla_Cost_of_Goods_Sold_2011-2025_TSLA.csv
│   ├── number_of_units.csv                        # Vehicle production counts
│   └── ... (supporting data files)
├── supply_chain_graph/
│   ├── tesla_supply_chain_graph_2024.pkl          # Directed NetworkX graph (weights attached)
├── data_preprocessing.py                          # Cleans/joins data → preprocessed_data/*.csv
├── network_graph.py                               # Builds & saves the two-layer graph + figure
├── helper_functions.py                            # Common loaders/utilities used by H1/H2/H3
├── H1.py                                          # Baseline MC model + Hypothesis 1 analysis
├── H2.py                                          # Supplier concentration (HHI) experiment
├── H3.py                                          # Common-shock/clustered-failure experiment
├── Live_demo.py                                   # Interactive animation of MC runs
└── README.md
```

- **Tesla_specific_data/**: houses all input data. The key output of preparation is
**preprocessed_data/component_weights_2024_percent.csv**, which normalizes **BEA 2024 $1 weights** to **Tesla’s 2024 COGS**, providing per-category dollar allocations used throughout the model.
- **supply_chain_graph/**: serialized NetworkX graph file. The graph pickle reflects the directed structure.
- **data_preprocessing.py**: performs schema cleanup, unit normalization, joins BEA categories with supplier mappings, scales BEA shares to 2024 COGS, and writes the **preprocessed CSV** used by downstream modules.
- **network_graph.py**: assembles a **two-layer** supply chain graph
**(Suppliers → Industries → Tesla)** with **weighted edges**, validates directionality, and saves the pickles. Also produces the illustrative network figure embedded in the README.
- **helper_functions.py**: shared helpers (graph loading & checks, COGS loaders, random-draw builders, convenience math) so H1/H2/H3 remain focused on experiment logic.
- **H1.py**: implements the **core Monte Carlo simulation model** and **Hypothesis 1** tests (identifies high-weight categories that drive expected shortfalls, reports averages and risk metrics).
- **H2.py**: runs the **HHI concentration experiment** reweights a chosen category’s supplier shares (control vs. diversified treatment), reuses common random numbers, and reports mean/tail differences with **paired t-tests** and **bootstrap p-values**.
- **H3.py**: evaluates **clustered/common shocks** adds an event-level severity to all suppliers in a targeted industry when the event triggers, and compares loss distributions to independent failures using the same paired testing framework.
- **Live_demo.py**: provides a **live, animated** view of the network and running-average shortfall across simulations, useful for presentations and sanity checks.

---

## Data Preparation
The data preparation (`data_preprocessing.py`) part transforms the actual data into a clean, simulation-ready representation of the EV supply chain with consistent categories, weights, and supplier mappings.

### Inputs
- **BEA Input–Output (`BEA_industry_to_commodity_updated_data.csv`)**: used to derive component-category weights per $1 of motor-vehicle output.
- **Public supplier mentions (`Supplier_to_BEA_commodity_mapping.csv`)**: used to label named suppliers and map suppliers to specific industry categories.
- **Firm filings for COGS (`Tesla_Cost_of_Goods_Sold_2011-2025_TSLA.csv`) and units (`number_of_units.csv`)**: used to scale category weights into USD and to compute implied units in simulation.

### Data Cleaning
- **Normalize category labels** to a fixed set of nine BEA industry categories (title-cased, trimmed, deduplicated).
- **Standardize supplier names** (strip punctuation/whitespace, unify obvious aliases).
- **Percent/fraction alignment**: all edge weights coerced into [0,1], any inputs stored in percent are divided by 100.
- **Missing/zero guards**: categories or suppliers with missing weights are dropped or set to zero and excluded from normalization.
- **Temporal alignment & normalization**: BEA data only available **for 2024** and expressed **per $1 of output**. A COGS series is available for **2011–2025**, the **2024** COGS is selected to align with BEA. BEA category weights are **normalized to sum to 1.0 (2024)** and then **scaled by 2024 total COGS** to compute `cogs_allocated_usd` per category.

### Category Weights (Layer 1)
- From BEA direct requirements, compute each category’s **share of COGS** for the EV production in year 2024 since the BEA data is only available for that year.
- **Scaling to dollars**, multiply shares by **COGS for the year of 2024** to obtain `cogs_allocated_usd` per category.

**Output file**:
`Tesla_specific_data/preprocessed_data/component_weights_2024_percent.csv`

**Columns include**:
- `component_industry_description`
- `weight_share_percent (category share × 100)`
- `cogs_allocated_usd (scaled by COGS)`

### Supplier Mapping (Layer 2)
- Map each **named supplier to exactly one industry category** (many suppliers per category are allowed).
- Within each category, assign **within-category splits** across suppliers that **sum to the category weight**. Here second layer true shares are unknown, so **randomized allocations** are used for exploration with a **fixed RNG seed** for **reproducibility**.

Input file:
`Tesla_specific_data/Supplier_to_BEA_commodity_mapping.csv`
(Each row: `Supplier`, `BEA Commodity Row`.)

### Quality & Reproducibility Checks
- **Totals**: Category shares sum to 100%. Supplier splits within each category sum to that category’s share.
- **Non-negativity**: all weights ≥ 0.
- **Determinism**: random allocations seeded to ensure **identical results across runs**.

---

## Supply Chain Network Graph

### Three-layer directed network 
The graph layout represents Tesla’s upstream structure as a **supplier → industry → Tesla** using NetworkX.

- **Layer 2 – Suppliers**: Named supplier firms (left). Each has `in-degree = 0` and edges into one or more industries.
- **Layer 1 – Industries**: **Nine BEA industry categories** (middle) that roll up the industry share in total EV production. Each industry has one outgoing edge to Tesla.
- **Layer 0 – Tesla**: Single sink node (right).

### Edge weights (two meanings, both coerced to [0,1])
- **Industry → Tesla**: BEA category shares of motor-vehicle COGS. Across all industries these **sum to 1.0**.
- **Supplier → Industry**: **Within-industry splits** across suppliers. For **proof-of-concept**, shares are drawn from a **Dirichlet distribution (seeded)** so that, per industry, supplier shares **sum to 1.0** and are reproducible.

### Directionality and storage 
The canonical file `supply_chain_graph/tesla_supply_chain_graph_2024.pkl` is created for storing the directed graph as a pickle file. Nodes are tagged with a `layer` attribute, edges carry a numeric `weight` and are oriented **supplier → industry → Tesla**.

This structure makes the interpretation of shocks unambiguous, a supplier failure reduces its industry’s available share, which in turn reduces Tesla’s effective COGS contribution for that category. With **validated direction and traceable weights**, the graph is a defensible basis for Monte Carlo analysis and hypothesis testing.

### Figure: Tesla two-layer supplier network with category weights.
<center> 
  <img width="581" height="497" alt="network graph" src="https://github.com/user-attachments/assets/dd240f7e-7786-4f75-bd28-233328cfaed8" />
</center>

---

## Monte Carlo Simulation Model
In this monte carlo simulation model, estimate how random supplier disruptions propagate to industry category availability and **total COGS** (and thus impact vehicle units). The model runs **10,000** independent trials and summarizes both average and tail outcomes.

### Model Setup
- **Nodes & edges**. A directed, two-layer graph: **Suppliers → Industries → Tesla**. Edges carry shares (in [0,1], percent inputs are coerced).
- **Category dollar base**. Each industry has an original **COGS allocation (USD)** for the analysis year (2024), derived from BEA category weights aligned to Tesla COGS.
- **Key parameters**
    - **Number of runs**: `NUM_SIMS` = 10,000
    - **Supplier failure probability per run**: `P_FAIL_BASE` = 0.05
    - **Failure severity (if failed)**: Uniform [0.30, 1.00] fraction of the supplier’s share
    - **COGS per vehicle**: `COGS_PER_VEHICLE_USD` = $45,245.32
    - **Baseline total COGS (2024)**: `TOTAL_COGS_REFERENCE` = $80.24B

### Random shock process (per run)
- **Draw failures**. For each supplier, sample a Bernoulli(p_fail).
- **Assign severity**. For failed suppliers only, **sample a severity** so that it fails partially, not completely, every time, `s ∈ [0.30,1.00]`, non-failed suppliers have `s = 0`.

### Availability and COGS propagation
For each **industry**:
- Start with full availability `avail = 1.0`.
- For each incoming supplier edge with share `w`, reduce availability by `w × s` from that supplier’s draw.
- Clamp: `avail = max(0, avail)`.
- Translate to dollars: value_left = industry_COGS_USD × avail.

**Total simulated COGS** for the run is the sum of `value_left` over all industries.
**Implied vehicle units** = `Total simulated COGS / COGS_per_vehicle`.
**Bottleneck industry** for that run is the industry with the **minimum availability**.

### Outputs & metrics
- **Per-run series**: total COGS, implied units, and bottleneck label.
- **Aggregates**: mean COGS, **average shortfall** vs. baseline, **p95 loss**, and the **frequency** each industry appears as the bottleneck.
- **Units impact** is reported as a percentage drop relative to a 2024 baseline units figure.

### Live demonstration (animation)
The live demo replays milestones across the 10,000 runs:
- **Top panel (network view)**:
    - **Highlights failed supplier → industry** edges in red with a severity label (e.g., “62%”).
    - Shows a summary box with **Run, Total COGS, No. of units, Impact on units**, and the **current bottleneck** industry.
- **Bottom panel (convergence)**:
  - Plots the **running average shortfall** (in $ billions) across runs.
  - An orange marker indicates the current milestone.

The simulation is linear in the number of edges per run (efficient), separates **stochastic shocks** from **deterministic propagation**, and produces interpretable metrics connecting supply failures to production-scale outcomes. The convergence plot provides a simple sanity check that 10,000 runs are sufficient for stable estimates.

### Figure: Live demo snapshot of the supply chain network and convergence curve
<center>
  <img width="1395" height="730" alt="mc_model_image" src="https://github.com/user-attachments/assets/affb81dd-b8e5-43b3-b85b-ba480b32b0d2" />
</center>

---

## Hypothesis 1

**H1 statement**: In Tesla’s supply network, the two industries with the largest baseline COGS weights are expected to drive **more than 50%** of total simulated loss when random supplier failures occur. If true, this would indicate **highly concentrated vulnerability**.

### Experimental design
- **Baseline by industry**. Each industry’s COGS allocation is fixed from the BEA-scaled 2024 shares.
- **Random supplier's disruption (10,000 runs)**. In every run, supplier nodes fail independently with probability `p` and receive a random **severity in [30%, 100%]** when failed.
- **Loss attribution**. For each run, the model computes:
    - remaining availability per industry,
    - **industry loss** (baseline COGS × severity passed through its suppliers), and
    - **share of total loss** contributed by each industry.
- **Top-2 share**. Industries are ranked by loss **within each run**, their combined share forms the **Top-2 share** for that run. The 10,000 values form the sampling distribution used for inference.

### Statistical test
- **Null (H₀)**: mean Top-2 loss share <= **0.50**.
- **Alternative (H₁)**: mean Top-2 loss share > **0.50 (dominance)**.
- **Estimator**: sample mean of the 10,000 Top-2 shares.
- **Test**: one-sample **t-test** of the mean against 0.50 (one-sided, “>”).
- **Decision rule**: if **t is large positive** and **p < 0.05**, conclude that the Top-2 industries dominate losses, otherwise, **do not support** H1.

**Interpretation** 
- **t-statistic** measures how far the observed mean is above 0.50 in standard-error units.
- **p-value** is the probability of seeing a mean this large (or larger) if the true mean were 0.50.
- A **large negative t** or **p ≈ 1** indicates the Top-2 share is **well below 50% (hypothesis not supported)**.

### Figure showcasing the actual printed test results
<center>
  <img width="842" height="524" alt="H1 results" src="https://github.com/user-attachments/assets/6dbd1e21-ca9f-48b1-b541-6b141f6fc490" />
</center>


The combined **Top-2 share was ~35% (< 50%)** with a strongly negative **t and p ≈ 1.0**, so **H1 is not supported losses are not dominated by just two industries under the independent-failure model**.

### Figure: Histogram of Top-2 share (10,000 runs)
The vertical dashed line marks the **50% threshold**, the red line marks the **sample mean** Top-2 share. Mass far **left of 50%** visually signals non-dominance.

<center>
  <img width="796" height="596" alt="plot 1" src="https://github.com/user-attachments/assets/f18c1b55-7c23-4a9e-ae0d-04ef11085a22" />
</center>

### Figure: Category loss bar chart
Bars show **average loss by industry ($ B)** across all runs, the highlighted overlay reports the **combined Top-2 share** of loss for context.

<center>
  <img width="799" height="599" alt="plot 2" src="https://github.com/user-attachments/assets/e09c040a-79e6-4635-a15c-8f165c37e041" />
</center>

---

## Hypothesis 2

**H2 statement**: Within a component category, spreading volume across more suppliers (i.e., lowering concentration/HHI) should **reduce both the average loss and the tail loss (p95)** compared with keeping one dominant supplier at the same total category weight.

### Experimental design
- **Control (concentrated)**: the same supplier set is kept, but one supplier is forced to **80% share** and the remaining suppliers split the **other 20% randomly**.
- **Treatment (diversified)**: the same supplier set is reweighted to **equal shares** (minimum HHI for a fixed set).
- **Everything else identical**: total category weight into Tesla, failure probability, severity distribution, graph topology.
- **Common random numbers**: both arms use the **same 10,000 shock paths**, so any difference is due to reweighting, not luck.

### Core measures and tests
- **Herfindahl–Hirschman Index (HHI)**: `HHI = ∑s_i^2` using fractional shares.
  - **Closer to 1 → concentrated** (e.g., 80/20 ≈ 0.68).
  - **Closer to 0 → diversified** (e.g., equal across 9 suppliers ≈ 0.11).
- **Mean-loss reduction**: compute per-run paired differences `d_r = Loss_control,r − Loss_treat,r`. Use a **paired t-test** on `d_r` (large-sample t; 95% CI). Here positive mean `d_r`
  with **p < 0.05** which concludes diversification lowers average loss.
- **Tail-loss reduction**: compute **p95(loss)** in each arm and take the difference. A **paired bootstrap** (resampling runs with replacement) gives a p-value. Here positive p95 difference with **p < 0.05** concludes diversification lowers tail risk.
- Hence H2 is supported for a category when both mean-loss and tail-loss tests are positive and significant.

### Focus Industry - `Computer and electronic products`
Over here the target industry is selected and the results associated with this is as shown in the below figure.

<center>
  <img width="424" height="324" alt="industry specific results" src="https://github.com/user-attachments/assets/f1a08f92-0c8f-491c-a187-654eaf2ffc5d" />
</center>

Despite a large HHI drop (0.65 → 0.11), neither the **average loss** nor the **p95 loss** improves significantly. The histogram overlay shows nearly overlapping loss distributions, and the QQ plot of paired differences shows many differences clustered near zero with only a few large positives—insufficient for significance. **Conclusion for this category: does not support Hypothesis 2**.

### Cross Industry results
Over here, the results were derived for each and every industry to get the industry-specific hypothesis testing.

<center>
  <img width="1378" height="251" alt="image" src="https://github.com/user-attachments/assets/90dc600d-5abf-4d5b-9a95-6ac7e9c881cf" />
</center>

The benefit of diversification depends on **how that category connects into the network** (its total weight into Tesla), the **number and balance of its suppliers**, and how often it becomes the **bottleneck** when shocks hit. In some categories (e.g., **Fabricated metal products**), equalizing shares **materially reduces extreme shortfalls**, producing significant **mean and tail** improvements. In others, losses are driven by **systemic effects or other categories**, so reweighting within the category has **limited impact**. Here the for the conclusion column, the decision was made on the basis of mean-loss reduction > 0 with p < 0.05 (paired t-test) and  p95-loss reduction > 0 with p < 0.05 (paired bootstrap).


### Figure: Histogram overlay plot — Control vs Treatment loss (Billions USD)
A side-by-side (overlaid) histogram of the **per-run total loss** for the two arms:
- **Blue** = concentrated control (80/20).
- **Orange** = diversified treatment (equal shares).
Each bar counts how many of the 10,000 runs produced a loss that falls inside that bin. If the orange mass consistently sits to the left of blue, treatment tends to lower losses, to the right means higher losses, on top of each other means no material difference.

**The analysis for the industry Computer and electronic products**
- The two histograms **almost coincide** across the support, including the right tail.
- There is **no visible bulk shift** of orange relative to blue.
- This matches the statistics: tiny mean-loss difference (~$0.003B) and non-significant p-value (~0.70).
- So for this category, equalizing supplier shares **did not materially change** the distribution of total losses under the current failure rate/severity and network bottlenecks.

Because the **entire distribution** (center and tail) is almost unchanged, there is **no distributional evidence** here that diversification reduces loss.

<center>
  <img width="596" height="543" alt="plot 1" src="https://github.com/user-attachments/assets/7051e0fa-62a3-4602-83e8-527d63f804e6" />
</center>

### Figure: QQ plot — Paired differences vs Normal(μ,σ)
A quantile–quantile (QQ) plot of the paired per-run differences 
`d_r = Loss_control,r − Loss_treat,r` against a reference **Normal(μ,σ)** with the same sample mean and standard deviation as `d_r`.

- **X-axis**: “theoretical” normal quantiles (what would be expected if `d_r` were perfectly normal).
- **Y-axis**: empirical quantiles of `d_r` in dollars.
- The **45° line** is the “perfect normal” benchmark.
    - Points on the line → data follow Normal(μ,σ).
    - Points **above** the line on the right → **heavier right tail** (some runs where control ≫ treatment, i.e., large positive benefit from diversification).
    - Points **below** the line on the left → **heavier left tail** (some runs where treatment is worse).
    - Flat **horizontal segments near 0 → many runs with very small differences** (mass at/near zero, because those nodes were never failed in the mc run).
- A **long, flat band near zero**: the vast majority of paired differences are **very small**, reweighting often does **almost nothing** because the bottleneck lies elsewhere or shocks don’t hit the dominant supplier hard enough.
- A **handful of large positive points** far above the line are **rare scenarios** where diversification helps a **lot** (big control-minus-treatment gains).
- Some modest negatives on the left, a **few** runs where diversification slightly **hurts**.

Overall, the distribution is **non-normal and highly skewed**, dominated by **near-zero differences** with **few large positives**—consistent with **a small mean** and **wide variance**, hence the **non-significant** paired t-test.

<center>
  <img width="583" height="539" alt="plot 2" src="https://github.com/user-attachments/assets/2a38d90c-5c4f-4810-a0a6-829caaef90d9" />
</center>

---

## Hypothesis 3

**H3 statement**: When several suppliers fail together due to a common event (e.g., shared sub-tier or region), production loss should rise materially versus independent failures, with a **marked increase in the p95 loss**.

Over here, **the scenario framed for this hypothesis** was there is *one regional shock hits the `Computer and electronics products` industry, so due to which multiple upstream suppliers to that industry are disrupted at once. In simulations, when this event occurs (with probability `pevent`), the whole cluster fails together with drawn severities, while all other suppliers across the network still fail independently as in the baseline. So now MC model predicts that this correlated cluster shock will push both the mean loss and especially the p95 loss 
markedly higher than the independent-failure baseline.*

### Experiment design
- **Cluster definition**. All supplier nodes feeding the `Computer and electronic products` industry (**9 suppliers** in the graph).
- **Two matched arms**.
    - **Independent baseline**: every supplier fails independently with probability `p_fail = 0.05`, severities `∼U[0.30,1.00]` applied to failed suppliers.
    - **Clustered-shock arm**: identical draws as baseline **plus a common event** that occurs with probability **p_event = 0.10**. When it occurs in run `r`, a shock `S_r ∼ U[0.30,0.60]` is **added** to the severities of **all cluster suppliers** (clamped to 1.0). All other suppliers in the network remain independent.
- **Paired comparison** The same random seeds are used for both arms to create **paired losses per run**, enabling a high-power **paired t-test** on the mean difference and a **paired bootstrap** test for the p95 difference.

### Key parameters
`Runs: 10,000 , p_fail=0.05 , p_event=0.10 , common shock S_r ∼ U[0.30,0.60]`

### Results
For the target specific industry `Computer and electronic products`, below figure shows the actual results.

<center>
  <img width="551" height="323" alt="h3 results" src="https://github.com/user-attachments/assets/d5a57bc4-657b-4cc4-b5a5-1d7dea37ced7" />
</center>

This test concludes that it **supports H3**. Both the average loss and the high-tail (p95) loss rise significantly when a common event simultaneously hits all suppliers in the targeted industry. When several suppliers fail together due to a common event (e.g., shared sub-tier or region), production loss should rise materially versus independent failures, with a **marked increase in the p95 loss**. 

### Figure: Histogram overlay (Loss Distributions, H3)
The **orange** clustered-shock distribution is visibly **right-shifted** relative to the **blue** independent baseline, with more mass at larger losses, evidence of higher expected loss and fatter tail under correlated failures.

<center>
  <img width="549" height="366" alt="plot 1" src="https://github.com/user-attachments/assets/8d85e9ac-5f37-43b0-a9a6-746baff4eb50" />
</center>

### Figure: QQ plot of paired differences
Points plot empirical quantiles of (cluster − independent) loss against a normal reference with the same mean/SD.
- The long **flat band near zero** reflects runs where no cluster event happened (differences ~0).
- The **upper-right arc well above the 45° line** captures event runs where the common shock pushes losses much higher, these positive extremes drive the significant t-statistic and the larger p95.

<center>
  <img width="536" height="375" alt="plot 2" src="https://github.com/user-attachments/assets/42574ab9-0310-4fbd-babb-9ed3c824af6d" />
</center

This leads to the conclusion that a **common (clustered) shock materially raises average loss and—especially—amplifies the high-loss tail (p95) versus independent failures**.

---

## Limitations
The primary limitation is **data availability**. Because detailed, firm-level bill-of-materials (BOM), supplier allocations, and sub-tier linkages are proprietary, the study is necessarily a **proof-of-concept** that demonstrates method and workflow rather than producing audited point estimates. The design is intentionally modular so it can be **re-run with real inputs** when they become available. 

Results should be interpreted as **directional risk signals** identifying where concentration, clustering, or high-weight categories could drive losses **not as audited forecasts**. With access to 
- **Company-specific BOM by category**
- **Actual supplier allocations and capacities**
- **sub-tier/region linkages**
- **empirically estimated failure/repair distributions**

The same code path can be re-run to produce **decision-grade** estimates and policy experiments.

---

## Dependencies

To run the simulation, install the following dependencies:

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- NetworkX
- SciPy
- Tkinter (required by the `TkAgg` Matplotlib backend used in the live animation window)

---

## Data Sources and References

- **Alsettevs public article (For list of suppliers)** –  
  [https://alsettevs.com/who-are-the-suppliers-of-tesla-parts/](https://alsettevs.com/who-are-the-suppliers-of-tesla-parts/)

- **BEA Input–Output Tables (Industry by Commodity)** - 
  [https://www.bea.gov/data/special-topics/input-output](https://www.bea.gov/data/special-topics/input-output)

- **Investopedia public article (For list of suppliers)** -
  [https://www.investopedia.com/ask/answers/052815/who-are-teslas-tsla-main-suppliers.asp](https://www.investopedia.com/ask/answers/052815/who-are-teslas-tsla-main-suppliers.asp)

- **SEC data (For COGS from Tesla’s annual reports)** - 
  [https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=TSLA&type=10-K&dateb=&owner=exclude&count=100](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=TSLA&type=10-K&dateb=&owner=exclude&count=100)

- **Tesla’s Statista Report (For No. of units)** - 
  [https://www-statista-com.proxy2.library.illinois.edu/study/23072/tesla-statista-dossier/](https://www-statista-com.proxy2.library.illinois.edu/study/23072/tesla-statista-dossier/)

- **Gupta, R., Li, J., & Fernandez, P. (2025).** *Evaluating risk factors in automotive supply chains: Implications for resilience.*  
  *Journal of Manufacturing Systems.* Elsevier.  
  [https://www.sciencedirect.com/science/article/pii/S2199853125000241](https://www.sciencedirect.com/science/article/pii/S2199853125000241)

- **López, A., Chen, H., & Nakamura, S. (2025).** *Modeling supply chain disruptions due to geopolitical risks.*  
  *Transportation Research Part E: Logistics and Transportation Review.* Elsevier.  
  [https://www.sciencedirect.com/science/article/pii/S136655452500331X](https://www.sciencedirect.com/science/article/pii/S136655452500331X)


