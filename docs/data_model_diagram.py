"""Generate clean, professional data model diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("white")


def draw_table(ax, x, y, title, columns, width=8.0):
    """Draw a database table. y is TOP of the table."""
    row_h = 0.38
    header_h = 0.50
    total_h = header_h + len(columns) * row_h + 0.15

    # Shadow
    shadow = patches.FancyBboxPatch(
        (x + 0.08, y - total_h - 0.08), width, total_h,
        boxstyle="round,pad=0.1", facecolor="#e2e8f0", edgecolor="none", zorder=0
    )
    ax.add_patch(shadow)

    # Main box
    box = patches.FancyBboxPatch(
        (x, y - total_h), width, total_h,
        boxstyle="round,pad=0.1", facecolor="white", edgecolor="#334155", linewidth=2, zorder=1
    )
    ax.add_patch(box)

    # Header
    header = patches.FancyBboxPatch(
        (x, y - header_h), width, header_h,
        boxstyle="round,pad=0.1", facecolor="#1e40af", edgecolor="#1e40af", linewidth=2, zorder=2
    )
    ax.add_patch(header)
    # Cover bottom corners of header
    ax.add_patch(patches.Rectangle((x, y - header_h), width, 0.15, facecolor="#1e40af", edgecolor="none", zorder=2))

    ax.text(x + width / 2, y - header_h / 2, title,
            ha="center", va="center", fontsize=13, fontweight="bold", color="white", zorder=3)

    # Columns
    for i, (name, dtype, is_pk, is_fk) in enumerate(columns):
        cy = y - header_h - 0.08 - (i + 0.5) * row_h

        # Alternating row background
        if i % 2 == 0:
            ax.add_patch(patches.Rectangle(
                (x + 0.05, cy - row_h / 2 + 0.02), width - 0.1, row_h - 0.04,
                facecolor="#f8fafc", edgecolor="none", zorder=1.5
            ))

        # Key badge
        if is_pk:
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.15, cy - 0.12), 0.5, 0.24,
                boxstyle="round,pad=0.05", facecolor="#fef2f2", edgecolor="#dc2626", linewidth=0.8, zorder=3
            ))
            ax.text(x + 0.4, cy, "PK", ha="center", va="center", fontsize=7, fontweight="bold", color="#dc2626", zorder=4)
            name_x = x + 0.8
        elif is_fk:
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.15, cy - 0.12), 0.5, 0.24,
                boxstyle="round,pad=0.05", facecolor="#eff6ff", edgecolor="#2563eb", linewidth=0.8, zorder=3
            ))
            ax.text(x + 0.4, cy, "FK", ha="center", va="center", fontsize=7, fontweight="bold", color="#2563eb", zorder=4)
            name_x = x + 0.8
        else:
            name_x = x + 0.3

        # Column name
        color = "#1e293b"
        weight = "bold" if (is_pk or is_fk) else "normal"
        ax.text(name_x, cy, name, va="center", fontsize=10, color=color, fontweight=weight, zorder=3)

        # Data type (right-aligned)
        ax.text(x + width - 0.25, cy, dtype, va="center", ha="right", fontsize=9, color="#94a3b8", zorder=3)

    return x, y - total_h, x + width, y


# ── Table 1: ura_transactions (top-left) ──
t1_cols = [
    ("id", "UUID", True, False),
    ("project", "TEXT NOT NULL", False, False),
    ("street", "TEXT", False, False),
    ("x, y", "FLOAT (SVY21)", False, False),
    ("market_segment", "TEXT", False, False),
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
draw_table(ax, 0.5, 13.2, "ura_transactions", t1_cols, width=8.5)

# ── Table 2: ec_features (bottom-left) ──
t2_cols = [
    ("id", "UUID", True, False),
    ("transaction_id", "UUID", False, True),
    ("project", "TEXT NOT NULL", False, False),
    ("district", "TEXT", False, False),
    ("area_sqm", "FLOAT", False, False),
    ("floor_mid", "INT", False, False),
    ("lease_commence_year", "INT", False, False),
    ("years_from_launch", "INT", False, False),
    ("price_psm", "FLOAT", False, False),
    ("price", "NUMERIC(15,2)", False, False),
    ("created_at", "TIMESTAMP", False, False),
]
draw_table(ax, 0.5, 5.8, "ec_features", t2_cols, width=8.5)

# ── Table 3: model_registry (top-right) ──
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
draw_table(ax, 11.0, 13.2, "model_registry", t3_cols, width=8.5)

# ── Table 4: prediction_logs (bottom-right) ──
t4_cols = [
    ("id", "UUID", True, False),
    ("model_version", "TEXT NOT NULL", False, False),
    ("input_features", "JSONB NOT NULL", False, False),
    ("predicted_price", "NUMERIC(15,2)", False, False),
    ("latency_ms", "FLOAT", False, False),
    ("created_at", "TIMESTAMP", False, False),
]
draw_table(ax, 11.0, 7.5, "prediction_logs", t4_cols, width=8.5)

# ── Relationships ──

# ura_transactions → ec_features
ax.annotate(
    "", xy=(4.75, 5.8), xytext=(4.75, 6.6),
    arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.5, connectionstyle="arc3,rad=0")
)
ax.text(5.0, 6.2, "1:1 via transaction_id", fontsize=10, color="#2563eb", fontstyle="italic")

# model_registry → prediction_logs
ax.annotate(
    "", xy=(15.25, 7.5), xytext=(15.25, 9.1),
    arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.5)
)
ax.text(15.6, 8.3, "1:N via model_version", fontsize=10, color="#2563eb", fontstyle="italic")

# Data flow: ura_transactions → model_registry
ax.annotate(
    "", xy=(9.2, 10.5), xytext=(10.8, 10.5),
    arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.5, linestyle="--")
)
ax.text(9.3, 10.9, "Feature Eng → Training", fontsize=11, color="#059669", fontweight="bold")

# Data flow: model → prediction_logs
ax.annotate(
    "", xy=(9.2, 4.8), xytext=(10.8, 4.8),
    arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.5, linestyle="--")
)
ax.text(9.2, 5.2, "Inference → Logging", fontsize=11, color="#059669", fontweight="bold")

# ── Title ──
ax.text(10, 14.0, "EC Price Predictor — Data Model",
        ha="center", fontsize=20, fontweight="bold", color="#0f172a")
ax.text(10, 13.6, "PostgreSQL 16  |  4 Tables  |  UUID Primary Keys  |  JSONB Metadata",
        ha="center", fontsize=12, color="#64748b")

# ── Legend ──
legend_y = 0.6
ax.add_patch(patches.FancyBboxPatch((1.0, legend_y - 0.15), 0.5, 0.3,
             boxstyle="round,pad=0.05", facecolor="#fef2f2", edgecolor="#dc2626", linewidth=0.8))
ax.text(1.25, legend_y, "PK", ha="center", va="center", fontsize=8, fontweight="bold", color="#dc2626")
ax.text(1.7, legend_y, "Primary Key", fontsize=10, va="center", color="#334155")

ax.add_patch(patches.FancyBboxPatch((4.0, legend_y - 0.15), 0.5, 0.3,
             boxstyle="round,pad=0.05", facecolor="#eff6ff", edgecolor="#2563eb", linewidth=0.8))
ax.text(4.25, legend_y, "FK", ha="center", va="center", fontsize=8, fontweight="bold", color="#2563eb")
ax.text(4.7, legend_y, "Foreign Key", fontsize=10, va="center", color="#334155")

ax.plot([7.0, 8.0], [legend_y, legend_y], color="#2563eb", lw=2.5)
ax.plot([8.0], [legend_y], marker=">", color="#2563eb", markersize=8)
ax.text(8.3, legend_y, "Relationship", fontsize=10, va="center", color="#334155")

ax.plot([11.0, 12.0], [legend_y, legend_y], "--", color="#059669", lw=2.5)
ax.plot([12.0], [legend_y], marker=">", color="#059669", markersize=8)
ax.text(12.3, legend_y, "Data Flow", fontsize=10, va="center", color="#334155")

plt.tight_layout(pad=0.5)
plt.savefig("docs/data_model.pdf", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("docs/data_model.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: docs/data_model.pdf, docs/data_model.png")
