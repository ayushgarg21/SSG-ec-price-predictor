"""Generate data model diagram as PDF."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")

fig.patch.set_facecolor("white")

def draw_table(ax, x, y, title, columns, w=5.5, row_h=0.32):
    h = (len(columns) + 1) * row_h + 0.2
    # Shadow
    ax.add_patch(plt.Rectangle((x+0.05, y-h-0.05), w, h, facecolor="#d1d5db", edgecolor="none", zorder=0))
    # Box
    ax.add_patch(plt.Rectangle((x, y-h), w, h, facecolor="white", edgecolor="#1e293b", linewidth=1.5, zorder=1))
    # Header
    ax.add_patch(plt.Rectangle((x, y-row_h-0.1), w, row_h+0.1, facecolor="#1e40af", edgecolor="#1e293b", linewidth=1.5, zorder=2))
    ax.text(x + w/2, y - row_h/2 - 0.02, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=3)

    for i, (col_name, col_type, is_pk, is_fk) in enumerate(columns):
        cy = y - (i + 1.5) * row_h - 0.1
        prefix = ""
        if is_pk:
            prefix = "PK "
        elif is_fk:
            prefix = "FK "

        color = "#dc2626" if is_pk else "#2563eb" if is_fk else "#374151"
        ax.text(x + 0.2, cy, f"{prefix}{col_name}", fontsize=8.5, va="center", color=color,
                fontweight="bold" if (is_pk or is_fk) else "normal", zorder=3)
        ax.text(x + w - 0.2, cy, col_type, fontsize=8, va="center", ha="right", color="#6b7280", zorder=3)

    return x, y-h, x+w, y

# Table 1: ura_transactions
t1_cols = [
    ("id", "UUID", True, False),
    ("project", "TEXT NOT NULL", False, False),
    ("street", "TEXT", False, False),
    ("x, y", "FLOAT (SVY21)", False, False),
    ("market_segment", "TEXT (CCR/RCR/OCR)", False, False),
    ("area", "FLOAT (sqm)", False, False),
    ("floor_range", "TEXT", False, False),
    ("contract_date", "TEXT (MMYY)", False, False),
    ("type_of_sale", "TEXT (1/2/3)", False, False),
    ("price", "NUMERIC(15,2)", False, False),
    ("property_type", "TEXT", False, False),
    ("district", "TEXT", False, False),
    ("tenure", "TEXT", False, False),
    ("nett_price", "NUMERIC(15,2)", False, False),
    ("ingested_at", "TIMESTAMP", False, False),
]
draw_table(ax, 0.5, 10.5, "ura_transactions", t1_cols, w=5.8)

# Table 2: ec_features
t2_cols = [
    ("id", "UUID", True, False),
    ("transaction_id", "UUID", False, True),
    ("project", "TEXT NOT NULL", False, False),
    ("district", "TEXT", False, False),
    ("area_sqm", "FLOAT", False, False),
    ("floor_mid", "INT", False, False),
    ("lease_commence_year", "INT", False, False),
    ("years_from_launch", "INT", False, False),
    ("type_of_sale", "TEXT", False, False),
    ("market_segment", "TEXT", False, False),
    ("price_psm", "FLOAT", False, False),
    ("price", "NUMERIC(15,2)", False, False),
    ("created_at", "TIMESTAMP", False, False),
]
draw_table(ax, 0.5, 5.0, "ec_features", t2_cols, w=5.8)

# Table 3: model_registry
t3_cols = [
    ("id", "UUID", True, False),
    ("model_name", "TEXT NOT NULL", False, False),
    ("model_version", "TEXT NOT NULL", False, False),
    ("algorithm", "TEXT", False, False),
    ("metrics", "JSONB", False, False),
    ("parameters", "JSONB", False, False),
    ("artifact_path", "TEXT", False, False),
    ("is_active", "BOOLEAN", False, False),
    ("created_at", "TIMESTAMP", False, False),
]
draw_table(ax, 9.5, 10.5, "model_registry", t3_cols, w=5.8)

# Table 4: prediction_logs
t4_cols = [
    ("id", "UUID", True, False),
    ("model_version", "TEXT NOT NULL", False, False),
    ("input_features", "JSONB NOT NULL", False, False),
    ("predicted_price", "NUMERIC(15,2)", False, False),
    ("latency_ms", "FLOAT", False, False),
    ("created_at", "TIMESTAMP", False, False),
]
draw_table(ax, 9.5, 6.5, "prediction_logs", t4_cols, w=5.8)

# Relationships
# ura_transactions -> ec_features (1:1 via transaction_id)
ax.annotate("", xy=(3.4, 5.0), xytext=(3.4, 5.4),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2))
ax.text(4.2, 5.2, "1:1 (transaction_id)", fontsize=8, color="#2563eb", style="italic")

# model_registry -> prediction_logs (1:N via model_version)
ax.annotate("", xy=(12.4, 6.5), xytext=(12.4, 7.1),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2))
ax.text(12.8, 6.8, "1:N (model_version)", fontsize=8, color="#2563eb", style="italic")

# Data flow arrows
ax.annotate("", xy=(6.5, 8.5), xytext=(9.3, 8.5),
            arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2, ls="--"))
ax.text(7.2, 8.8, "Feature Engineering", fontsize=9, color="#059669", fontweight="bold")

ax.annotate("", xy=(6.5, 3.5), xytext=(9.3, 3.5),
            arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2, ls="--"))
ax.text(7.0, 3.8, "Model Inference → Logging", fontsize=9, color="#059669", fontweight="bold")

# Title
ax.text(8, 10.9, "EC Price Predictor — Data Model", ha="center", fontsize=16, fontweight="bold", color="#1e293b")
ax.text(8, 10.6, "PostgreSQL 16 | 4 Tables | UUID Primary Keys | JSONB for Flexible Metadata",
        ha="center", fontsize=10, color="#6b7280")

# Legend
legend_y = 0.8
ax.text(0.5, legend_y, "Legend:", fontsize=9, fontweight="bold", color="#374151")
ax.add_patch(plt.Rectangle((2.0, legend_y-0.12), 0.3, 0.24, facecolor="#dc2626", alpha=0.2))
ax.text(2.5, legend_y, "PK = Primary Key", fontsize=8, color="#dc2626")
ax.add_patch(plt.Rectangle((5.0, legend_y-0.12), 0.3, 0.24, facecolor="#2563eb", alpha=0.2))
ax.text(5.5, legend_y, "FK = Foreign Key", fontsize=8, color="#2563eb")
ax.plot([8.0, 8.8], [legend_y, legend_y], "--", color="#059669", lw=2)
ax.text(9.0, legend_y, "Data Flow", fontsize=8, color="#059669")

plt.tight_layout()
plt.savefig("docs/data_model.pdf", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("docs/data_model.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: docs/data_model.pdf, docs/data_model.png")
