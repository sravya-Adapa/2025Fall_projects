# IS-597 PR project work
# Name: Dhyey Kasundra
# Netid: dk65
# Date: 11-27-2025

# Teammate Name: Sravya Adapa
# Net Id: nadapa2

#Importing necessary libraries
import re
import numpy as np
import pandas as pd
from pathlib import Path


# Parse year from strings like "Q1 2024"
def parse_year_q():
    """
    This function will parse the year, quarter from the input string. It extracts 4 digit year lookinf like 20xx
    and quarter [Q1,Q2,Q3,Q4,q1,q2,q3,q4]. It returns the quarter, year in the iput string.

    :param s: input string
    :return: year, quarter

    >>> parse_year_q("2023 Q1")
    (2023, 1)

    >>> parse_year_q("2021 q4")
    (2021, 4)

    >>> parse_year_q("Q2 2022 Earnings")
    (2022, 2)

    >>> parse_year_q("2020")
    (2020, nan)

    >>> parse_year_q("Q3")
    (nan, 3)
    """
    s = str(s)
    m_y = re.search(r'(20\d{2})', s)
    yr  = int(m_y.group(1)) if m_y else np.nan
    m_q = re.search(r'[Qq]\s*([1-4])', s)
    qt  = int(m_q.group(1)) if m_q else np.nan
    return yr, qt


if __name__ == "__main__":

    """ Here initially the code reads two input csv files that is Tesla_Cost_of_Goods_Sold and number of units produced.
    From this aggregate sums 2024 quarterly COGS (values in billions USD) and 2024 units, and also computed COGS per vehicle.
    """

    COGS_PATH = Path("Tesla_specific_data/Tesla_Cost_of_Goods_Sold_2011-2025_TSLA.csv")
    UNITS_PATH = Path("Tesla_specific_data/number_of_units.csv")

    OUT_SCALE_PATH = Path("Tesla_specific_data/preprocessed_data/scale_2024.csv")  # contains year, cogs_2024_usd, units_2024, cogs_per_vehicle_usd
    OUT_COGS = Path("Tesla_specific_data/preprocessed_data/tesla_cogs_2024_quarterly_usd.csv")  # 2024 COGS quarters with an added COGS_USD (in dollars)
    OUT_UNITS = Path("Tesla_specific_data/preprocessed_data/tesla_units_2024_quarterly.csv")  # 2024 units rows as-is (for traceability)

    # Read the COGS file
    cogs = pd.read_csv(COGS_PATH)
    cogs["Date"] = pd.to_datetime(cogs["Date"], errors="coerce")

    # Keep 2024 rows, convert Quarterly (billions) to USD for easy computation
    cogs_2024 = cogs[cogs["Date"].dt.year == 2024].copy()
    cogs_2024["COGS_USD"] = pd.to_numeric(cogs_2024["Quarterly"], errors="coerce") * 1_000_000_000
    cogs_2024_sum = float(cogs_2024["COGS_USD"].sum())

    # Read the UNITS file
    units_raw = pd.read_csv(UNITS_PATH)

    # If headers are generic, rename them
    if units_raw.shape[1] < 2:
        raise ValueError("Units file must have at least two columns: period and value.")

    period_col = units_raw.columns[0]
    value_col = units_raw.columns[1]

    units = units_raw[[period_col, value_col]].copy().rename(
        columns={period_col: "Period", value_col: "Units"}
    )

    years, qtrs = zip(*[parse_year_q(v) for v in units["Period"]])
    units["_year"] = years
    units["_quarter"] = qtrs

    # Clean numeric (remove commas) and keep 2024
    units["Units"] = pd.to_numeric(units["Units"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    units_2024 = units[units["_year"] == 2024].copy()
    units_2024_sum = float(units_2024["Units"].sum())

    # SCALE + SAVES the final output file
    cogs_per_vehicle = round((cogs_2024_sum / units_2024_sum),2) if units_2024_sum > 0 else float("nan")

    scale = pd.DataFrame({
        "year": [2024],
        "cogs_2024_usd": [cogs_2024_sum],
        "units_2024": [units_2024_sum],
        "cogs_per_vehicle_usd": [cogs_per_vehicle],
    })

    scale.to_csv(OUT_SCALE_PATH, index=False)
    cogs_2024.to_csv(OUT_COGS, index=False)
    units_2024.to_csv(OUT_UNITS, index=False)

    print("Wrote:")
    print(" -", OUT_SCALE_PATH)
    print(" -", OUT_COGS)
    print(" -", OUT_UNITS)

    """ Now derive the component-category weights from the BEA which will serves as an input file, over here this file is modified by
    Keeping only positive entries and excluding the self row (“Motor vehicles …” as a commodity). And after that only commodities 
    were include whose supplier list data was available to us.
    
    Over here That “Total industry output requirement” being bigger than 1$ simply means we’re looking at gross output, 
    not a bill-of-materials cost. In the Total Requirements table BEA applies the Leontief inverse, which captures all 
    upstream rounds of production: $1 of final demand for an industry triggers purchases of parts, which trigger purchases 
    of inputs to make those parts, and so on (including self-use inside the same industry). When you sum those cascading 
    requirements, the economy must produce more than $1 of gross output across all industries to deliver $1 of final demand 
    for the target industry. It’s an intensity/multiplier, not a cost share, and that’s why the total exceeds one.
    
    Steps followed were:
    1. Kept only the required final rows (all positive) out of the whole BEA table.
    2. Normalized each row so the shares sum to 100%.
    3. Finally, mapped those shares to 2024 COGS total = $80.24B (from scale_2024.csv) 
    to get a dollar allocation by category. This gives you clean component weights which we can use 
    as the EV→category edge weights in the graph.
    """

    # Input File Paths
    bea_table_path = ["Tesla_specific_data/BEA_industry_to_commodity_updated_data.csv"]
    cogs_scale_path = ["Tesla_specific_data/preprocessed_data/scale_2024.csv"]

    # Load BEA
    bea_path = None
    for p in bea_table_path:
        if Path(p).exists():
            bea_path = p
            break
    if bea_path is None:
        raise FileNotFoundError("BEA file not found in expected locations.")

    bea = pd.read_csv(bea_path)
    scale = pd.read_csv(cogs_scale_path[0])

    # Determine descriptor and numeric column
    desc_col = bea.columns[0]
    name_col = bea.columns[1]

    # Coerce only true numeric candidate columns (skip desc_col & name_col)
    for c in bea.columns:
        if c not in (desc_col, name_col):
            bea[c] = pd.to_numeric(bea[c], errors="coerce")

    numeric_cols = [c for c in bea.columns if c not in (desc_col, name_col) and bea[c].notna().sum() > 0]
    target_col = max(numeric_cols, key=lambda c: bea[c].notna().sum())

    # Build, rename, and clean (drop NaN only for the numeric 'value' column)
    df = bea[[desc_col, name_col, target_col]].copy()
    df = df.rename(columns={
        desc_col: "component_Category",
        name_col: "component_industry_description",
        target_col: "value"
    })
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0]

    total = df["value"].sum()
    df["weight_share"] = df["value"] / total
    df["weight_share_percent"] = round((df["weight_share"] * 100),2)

    # COGS allocation
    cogs_total = float(scale.loc[scale["year"] == 2024, "cogs_2024_usd"].iloc[0])
    df["cogs_allocated_usd"] = round((df["weight_share"] * cogs_total),2)

    df = df.sort_values("weight_share", ascending=False).reset_index(drop=True)

    out_path = "Tesla_specific_data/preprocessed_data/component_weights_2024_percent.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)




