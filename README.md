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

## Project Description
This project simulates how disruptions in a **giant electric vehicle manufacturing firm like Tesla**, in their supplier network affect it’s ability to **produce vehicles and control production cost**.

An EV manufacturer depends on key component families like **battery systems, power electronics, motors/drive units, body/metals, interior/trim, plastics/rubber, and electronics**. When supply shocks occur due to factory outages, quality recalls, logistics delays, or regional events inputs from critical suppliers can drop suddenly. These disruptions constrain component availability at the plant and reduce finished-vehicle.

The Monte Carlo simulation combines:
- **BEA Input–Output (Direct Requirements)** — to estimate **component-category weights** per $1 of motor-vehicle output as an proxy to “BOM (Bill of Materials) by category”) since the EV manufacturing company’s BOM data is proprietary.  
- **Public supplier lists (articles)** — to derive a rough idea about the companies who contribute to Tesla’s EV production, and **map supplier names to component categories** for labeling the network.
- **Firm public filings (units/COGS)** — Tesla’s quarterly financial reports provide data related to **number of EV quantities and cost of goods sold**, to scale results to an EV production context.

The supply chain network is **two-layered** with:
- **Layer 1 (BOM by category)**: the EV product node connects to component-category nodes, **edge weights** are the BEA-derived shares.
- **Layer 2 (named suppliers)**: each category node connects to its known multiple suppliers whose names are derived from the publicly available articles, **within-category splits** are clearly marked assumptions (randomized allocations that sum to the category weight, used only for proof-of-concept exploration).

Monte Carlo simulations introduce random disruptions to these supplier nodes across **10,000 runs**, capturing how failures in one or multiple suppliers propagate through the network. The model produces a **probabilistic assessment of production vulnerability**, identifying which component categories and supplier groupings contribute most to potential production shortfalls.

A **NetworkX-based visualization** will illustrate the firm’s supplier network, EV product to component categories (weighted edges), then categories to labeled suppliers (assumed splits), highlighting **high-impact links** where simulated disruptions have the largest effect on production.


## Hypotheses

**H1:** Failures in high-weight component categories (battery systems, electronics, primary metals) will drive most of the simulated production loss. In the baseline, the top **two categories** will account for the **majority** of expected production shortfalls.

**H2:** Within a component category, **spreading volume across more suppliers** (lower concentration/HHI) will reduce both **average loss and tail risk** compared with keeping one dominant supplier at the same total category weight.

**H3:** When **several suppliers fail together** due to a common event (same region or shared sub-tier), the simulated production loss will be **much higher** than under independent failures, with p95 loss rising sharply.

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

### Figure: Live demo snapshot of the supply chain network and convergence curve.
<center>
  <img width="1395" height="730" alt="mc_model_image" src="https://github.com/user-attachments/assets/affb81dd-b8e5-43b3-b85b-ba480b32b0d2" />
</center>

---

## Hypothesis 1

---

## Hypothesis 2

---

## Hypothesis 3

---

## Limitations

---

## Dependencies

To run the simulation, install the following dependencies:

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- NetworkX

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


