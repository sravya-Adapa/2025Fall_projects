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
- [Data Preparation](#data-preparation)
- [Repository Structure](#repository-structure)
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

---

## Repository Structure

---

## Supply Chain Network Graph

---

## Monte Carlo Simulation Model

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


