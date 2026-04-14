"""
Executive Condominium Price Analysis — Exploratory Data Analysis
================================================================
This script produces all EDA visualisations and statistical summaries
used to inform the feature engineering and modelling decisions.

Run: python notebooks/01_eda.py
Output: notebooks/figures/
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from scipy import stats

from src.features.engineering import parse_contract_date, extract_lease_commence_year, extract_floor_mid

# ── Setup ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

engine = create_engine("postgresql://ec_user:ec_password@localhost:5432/ec_prices")

print("Loading data...")
with engine.connect() as conn:
    df = pd.read_sql(text("SELECT * FROM ura_transactions WHERE property_type = 'Executive Condominium'"), conn)

print(f"EC transactions loaded: {len(df):,}")

# ── Parse dates and derived columns ──────────────────────────────────
parsed = df["contract_date"].apply(lambda x: pd.Series(parse_contract_date(x)))
df["txn_month"] = parsed[0]
df["txn_year"] = parsed[1]
df["floor_mid"] = df["floor_range"].apply(extract_floor_mid)
df["lease_commence_year"] = df["tenure"].apply(extract_lease_commence_year)
df["years_from_launch"] = df["txn_year"] - df["lease_commence_year"]
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df["price_psm"] = df["price"] / df["area"]
df["district_num"] = pd.to_numeric(df["district"], errors="coerce")

# Sale type mapping
sale_map = {"1": "New Sale", "2": "Sub Sale", "3": "Resale"}
df["sale_type_label"] = df["type_of_sale"].map(sale_map)

clean = df.dropna(subset=["price", "area", "txn_year", "lease_commence_year", "floor_mid"])
print(f"Clean rows: {len(clean):,}")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Price Distribution
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(clean["price"] / 1e6, bins=50, edgecolor="white", alpha=0.8, color="#2563eb")
axes[0].set_xlabel("Price (S$ Million)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of EC Transaction Prices")
axes[0].axvline(clean["price"].median() / 1e6, color="red", ls="--", label=f'Median: S${clean["price"].median()/1e6:.2f}M')
axes[0].legend()

axes[1].hist(clean["price_psm"], bins=50, edgecolor="white", alpha=0.8, color="#059669")
axes[1].set_xlabel("Price per sqm (S$)")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of EC Price PSM")
axes[1].axvline(clean["price_psm"].median(), color="red", ls="--", label=f'Median: S${clean["price_psm"].median():,.0f}/sqm')
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_price_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_price_distribution.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Price Trends Over Time
# ══════════════════════════════════════════════════════════════════════
# Create year-quarter
clean["yq"] = clean["txn_year"].astype(int).astype(str) + "Q" + ((clean["txn_month"] - 1) // 3 + 1).astype(int).astype(str)

trend = clean.groupby("yq").agg(
    median_price=("price", "median"),
    median_psm=("price_psm", "median"),
    volume=("price", "count"),
).reset_index().sort_values("yq")

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(trend["yq"], trend["median_psm"], "o-", color="#2563eb", linewidth=2, markersize=4, label="Median PSM")
ax1.set_xlabel("Quarter")
ax1.set_ylabel("Median Price PSM (S$)", color="#2563eb")
ax1.tick_params(axis="y", labelcolor="#2563eb")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

ax2 = ax1.twinx()
ax2.bar(trend["yq"], trend["volume"], alpha=0.3, color="#f59e0b", label="Transaction Volume")
ax2.set_ylabel("Transaction Volume", color="#f59e0b")
ax2.tick_params(axis="y", labelcolor="#f59e0b")

fig.suptitle("EC Price Trends and Transaction Volume Over Time", fontsize=14, fontweight="bold")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_price_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_price_trends.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Price by District
# ══════════════════════════════════════════════════════════════════════
district_stats = clean.groupby("district_num").agg(
    median_price=("price", "median"),
    count=("price", "count"),
    median_psm=("price_psm", "median"),
).reset_index().sort_values("median_psm", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(district_stats["district_num"].astype(str), district_stats["median_psm"], color="#2563eb", alpha=0.8)
ax.set_xlabel("Median Price PSM (S$)")
ax.set_ylabel("Postal District")
ax.set_title("EC Median Price PSM by Postal District")
for bar, count in zip(bars, district_stats["count"]):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, f'n={count}', va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_price_by_district.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_price_by_district.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Price vs Years from Launch (THE KEY RELATIONSHIP)
# ══════════════════════════════════════════════════════════════════════
valid_years = clean[(clean["years_from_launch"] >= 0) & (clean["years_from_launch"] <= 30)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plot
year_groups = valid_years.groupby("years_from_launch")
bp_data = [group["price_psm"].values for name, group in year_groups if len(group) >= 10]
bp_labels = [name for name, group in year_groups if len(group) >= 10]
axes[0].boxplot(bp_data, labels=[str(int(l)) for l in bp_labels], showfliers=False)
axes[0].set_xlabel("Years from Lease Commencement")
axes[0].set_ylabel("Price PSM (S$)")
axes[0].set_title("Price PSM Distribution by Years from Launch")
axes[0].axvline(bp_labels.index(5) + 1 if 5 in bp_labels else 0, color="red", ls="--", alpha=0.5, label="MOP (5yr)")
axes[0].axvline(bp_labels.index(10) + 1 if 10 in bp_labels else 0, color="green", ls="--", alpha=0.5, label="Privatised (10yr)")
axes[0].legend()

# Median trend with confidence interval
yearly_median = valid_years.groupby("years_from_launch").agg(
    median=("price_psm", "median"),
    q25=("price_psm", lambda x: x.quantile(0.25)),
    q75=("price_psm", lambda x: x.quantile(0.75)),
    count=("price_psm", "count"),
).reset_index()
yearly_median = yearly_median[yearly_median["count"] >= 10]

axes[1].plot(yearly_median["years_from_launch"], yearly_median["median"], "o-", color="#2563eb", linewidth=2, label="Median PSM")
axes[1].fill_between(yearly_median["years_from_launch"], yearly_median["q25"], yearly_median["q75"], alpha=0.2, color="#2563eb", label="IQR")
axes[1].axvline(5, color="red", ls="--", alpha=0.7, label="MOP (5yr)")
axes[1].axvline(10, color="green", ls="--", alpha=0.7, label="Privatised (10yr)")
axes[1].set_xlabel("Years from Lease Commencement")
axes[1].set_ylabel("Price PSM (S$)")
axes[1].set_title("EC Price Trajectory Over Lease Life")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_price_vs_years.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_price_vs_years.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: New Sale vs Resale Price Comparison
# ══════════════════════════════════════════════════════════════════════
sale_data = clean[clean["sale_type_label"].isin(["New Sale", "Resale"])]

fig, ax = plt.subplots(figsize=(10, 6))
for label, color in [("New Sale", "#2563eb"), ("Resale", "#dc2626")]:
    subset = sale_data[sale_data["sale_type_label"] == label]
    yearly = subset.groupby("txn_year")["price_psm"].median().reset_index()
    ax.plot(yearly["txn_year"], yearly["price_psm"], "o-", color=color, linewidth=2, label=f"{label} (n={len(subset):,})")
ax.set_xlabel("Year")
ax.set_ylabel("Median Price PSM (S$)")
ax.set_title("New Sale vs Resale EC Prices Over Time")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_newsale_vs_resale.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_newsale_vs_resale.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════
numeric_cols = ["price", "area", "floor_mid", "district_num", "lease_commence_year",
                "years_from_launch", "txn_year", "price_psm"]
corr = clean[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Floor Level vs Price
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
floor_stats = clean.groupby("floor_mid")["price_psm"].agg(["median", "count"]).reset_index()
floor_stats = floor_stats[floor_stats["count"] >= 20]
ax.scatter(floor_stats["floor_mid"], floor_stats["median"], s=floor_stats["count"], alpha=0.6, color="#2563eb")
z = np.polyfit(floor_stats["floor_mid"], floor_stats["median"], 2)
p = np.poly1d(z)
x_line = np.linspace(floor_stats["floor_mid"].min(), floor_stats["floor_mid"].max(), 100)
ax.plot(x_line, p(x_line), "--", color="red", label=f"Quadratic fit")
ax.set_xlabel("Floor Level (midpoint)")
ax.set_ylabel("Median Price PSM (S$)")
ax.set_title("Floor Level Premium in EC Transactions")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_floor_vs_price.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_floor_vs_price.png")

# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Unit Size Distribution and Price Relationship
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(clean["area"], bins=50, edgecolor="white", alpha=0.8, color="#7c3aed")
axes[0].set_xlabel("Floor Area (sqm)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of EC Unit Sizes")

axes[1].hexbin(clean["area"], clean["price_psm"], gridsize=30, cmap="YlOrRd", mincnt=5)
axes[1].set_xlabel("Floor Area (sqm)")
axes[1].set_ylabel("Price PSM (S$)")
axes[1].set_title("Unit Size vs Price PSM (Density)")
plt.colorbar(axes[1].collections[0], ax=axes[1], label="Count")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_unit_size.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 08_unit_size.png")

# ══════════════════════════════════════════════════════════════════════
# STATISTICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)
print(f"\nDataset: {len(clean):,} clean EC transactions")
print(f"Date range: {int(clean['txn_year'].min())} — {int(clean['txn_year'].max())}")
print(f"Districts: {sorted(clean['district_num'].dropna().unique().astype(int).tolist())}")
print(f"Projects: {clean['project'].nunique()}")

print(f"\nPrice Statistics:")
print(f"  Median: S${clean['price'].median():,.0f}")
print(f"  Mean:   S${clean['price'].mean():,.0f}")
print(f"  Std:    S${clean['price'].std():,.0f}")
print(f"  Min:    S${clean['price'].min():,.0f}")
print(f"  Max:    S${clean['price'].max():,.0f}")

print(f"\nPrice PSM Statistics:")
print(f"  Median: S${clean['price_psm'].median():,.0f}/sqm")
print(f"  Mean:   S${clean['price_psm'].mean():,.0f}/sqm")

# MOP vs Privatised comparison
mop = clean[clean["years_from_launch"] == 5]["price_psm"]
priv = clean[clean["years_from_launch"] == 10]["price_psm"]
if len(mop) > 10 and len(priv) > 10:
    t_stat, p_val = stats.ttest_ind(mop, priv)
    print(f"\nMOP (5yr) vs Privatised (10yr):")
    print(f"  MOP median PSM:        S${mop.median():,.0f}")
    print(f"  Privatised median PSM: S${priv.median():,.0f}")
    print(f"  Difference:            S${priv.median() - mop.median():,.0f} ({(priv.median() - mop.median()) / mop.median() * 100:.1f}%)")
    print(f"  T-test p-value:        {p_val:.4e} ({'significant' if p_val < 0.05 else 'not significant'})")

# Sale type breakdown
print(f"\nSale Type Breakdown:")
for st in ["New Sale", "Resale", "Sub Sale"]:
    subset = clean[clean["sale_type_label"] == st]
    if len(subset) > 0:
        print(f"  {st}: {len(subset):,} txns, median S${subset['price'].median():,.0f}")

print(f"\n8 figures saved to {FIG_DIR}/")
