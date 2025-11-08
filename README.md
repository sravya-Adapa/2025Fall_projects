# Monte Carlo Simulation of Global Supply Chain Disruptions and Their Impact on U.S. Automotive Exports

## Team Members
- **Dhyey Kasundra**  
- **Sravya Adapa**

## Project Type
**Type II – Original Monte Carlo Simulation**

---

## Project Description
This project simulates how disruptions in the international trade of automotive parts and raw materials affect the **U.S. automotive industry’s ability to produce and export vehicles**.

The automotive sector depends heavily on imported inputs such as **steel, aluminum, semiconductors, batteries, plastics, and rubber**. When global supply chain shocks occur—such as **port shutdowns, geopolitical tensions, or natural disasters**—imports from key supplier countries can decline sharply. These disruptions can constrain the supply of essential inputs to U.S. manufacturing plants and ultimately **reduce U.S. automotive export capacity**.

The Monte Carlo simulation combines:
- **UN Comtrade trade data** – to quantify how much the U.S. relies on each supplier country for key materials.  
- **World Bank indices** – logistics performance and political stability indicators, used to estimate each country’s probability of trade disruption.  

Each supplier country is represented as a **node** connected to a **U.S. automotive manufacturing node**, with:
- **Edge weights** proportional to import shares, and  
- **Node attributes** derived from reliability/risk indicators.

Monte Carlo simulations introduce random disruptions to these supplier nodes across **10,000 runs**, capturing how failures in one or multiple countries propagate through the network. The model produces a **probabilistic assessment of export vulnerability**, identifying which materials or countries contribute most to potential production shortfalls.

A **NetworkX-based visualization** will illustrate the global supplier network, highlighting **high-risk connections** where disruptions have the largest simulated impact.

---

## Hypotheses

**H1:** Disruptions in imports of high-weight inputs such as steel and semiconductors will cause disproportionately higher reductions in simulated U.S. automotive export capacity compared to disruptions in lower-weight materials such as rubber and plastics.

**H2:** Increasing the number of supplier countries per input category (i.e., more diversified sourcing) reduces expected export loss by at least **25–30%** relative to concentrated sourcing scenarios.

**H3:** When correlated disruptions occur among high-risk supplier countries in the **East Asian region** (e.g., China, Japan, South Korea, Taiwan), total simulated export losses will exceed independent-shock predictions by at least **20–25%**, indicating **systemic regional vulnerability**.

---

## 📂 GitHub Repository
[https://github.com/sravya-Adapa/2025Spring_projects](https://github.com/sravya-Adapa/2025Spring_projects)

---

## 📊 Data Sources & References

- **UN Comtrade Database** – Bilateral U.S. trade data for HS 8701–8708  
  [https://comtradeplus.un.org/](https://comtradeplus.un.org/)

- **World Bank Logistics Performance Index & Political Risk Indicators**  
  [https://data.worldbank.org/](https://data.worldbank.org/)

- **BEA Input–Output Tables (Industry by Commodity)**  
  [https://www.bea.gov/data/special-topics/input-output](https://www.bea.gov/data/special-topics/input-output)

- **Gupta, R., Li, J., & Fernandez, P. (2025).** *Evaluating risk factors in automotive supply chains: Implications for resilience.*  
  *Journal of Manufacturing Systems.* Elsevier.  
  [https://www.sciencedirect.com/science/article/pii/S2199853125000241](https://www.sciencedirect.com/science/article/pii/S2199853125000241)

- **López, A., Chen, H., & Nakamura, S. (2025).** *Modeling supply chain disruptions due to geopolitical risks.*  
  *Transportation Research Part E: Logistics and Transportation Review.* Elsevier.  
  [https://www.sciencedirect.com/science/article/pii/S136655452500331X](https://www.sciencedirect.com/science/article/pii/S136655452500331X)

---

## Requirements

To run the simulation, install the following dependencies:

Python 3.8+
NumPy
Pandas
Matplotlib / Seaborn
NetworkX
