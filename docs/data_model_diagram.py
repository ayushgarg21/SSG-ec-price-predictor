"""Generate clean data model diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(24, 16))
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis("off")
fig.patch.set_facecolor("#fafafa")

# Title
ax.text(12, 15.5, "EC Price Predictor — Data Model",
        ha="center", fontsize=24, fontweight="bold", color="#0f172a")
ax.text(12, 15.0, "PostgreSQL 16  |  4 Tables  |  UUID Primary Keys  |  JSONB Metadata",
        ha="center", fontsize=13, color="#64748b")


def draw_table(ax, x, y, title, columns, width=9.5):
    """Draw a table. x,y = top-left corner of header."""
    row_h = 0.42
    header_h = 0.55
    padding = 0.12
    total_h = header_h + len(columns) * row_h + padding * 2

    # Shadow
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.1, y - total_h + 0.1), width, total_h - 0.2,
        boxstyle="round,pad=0.12", facecolor="#cbd5e1", edgecolor="none", zorder=0
    ))

    # Body
    ax.add_patch(patches.FancyBboxPatch(
        (x, y - total_h), width, total_h,
        boxstyle="round,pad=0.12", facecolor="white", edgecolor="#94a3b8", linewidth=1.5, zorder=1
    ))

    # Header
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.02, y - header_h + 0.02), width - 0.04, header_h,
        boxstyle="round,pad=0.1", facecolor="#1e40af", edgecolor="#1e40af", linewidth=0, zorder=2
    ))
    ax.add_patch(patches.Rectangle(
        (x + 0.02, y - header_h + 0.02), width - 0.04, 0.18,
        facecolor="#1e40af", edgecolor="none", zorder=2
    ))
    ax.text(x + width / 2, y - header_h / 2 + 0.02, title,
            ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=3,
            fontfamily="monospace")

    for i, (name, dtype, is_pk, is_fk) in enumerate(columns):
        cy = y - header_h - padding - (i + 0.5) * row_h

        if i % 2 == 0:
            ax.add_patch(patches.Rectangle(
                (x + 0.1, cy - row_h / 2 + 0.02), width - 0.2, row_h - 0.04,
                facecolor="#f1f5f9", edgecolor="none", zorder=1.5
            ))

        name_x = x + 0.35
        if is_pk:
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.2, cy - 0.13), 0.55, 0.26,
                boxstyle="round,pad=0.04", facecolor="#fef2f2", edgecolor="#ef4444", linewidth=1, zorder=3
            ))
            ax.text(x + 0.475, cy, "PK", ha="center", va="center", fontsize=8, fontweight="bold", color="#dc2626", zorder=4)
            name_x = x + 0.9
        elif is_fk:
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.2, cy - 0.13), 0.55, 0.26,
                boxstyle="round,pad=0.04", facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1, zorder=3
            ))
            ax.text(x + 0.475, cy, "FK", ha="center", va="center", fontsize=8, fontweight="bold", color="#2563eb", zorder=4)
            name_x = x + 0.9

        ax.text(name_x, cy, name, va="center", fontsize=11,
                color="#0f172a", fontweight="bold" if (is_pk or is_fk) else "normal", zorder=3)
        ax.text(x + width - 0.3, cy, dtype, va="center", ha="right",
                fontsize=10, color="#94a3b8", zorder=3)

    bottom_y = y - total_h
    return x, bottom_y, x + width, y


# ── TABLES ──

# ura_transactions (top-left)
t1 = [
    ("id", "UUID", True, False),
    ("project", "TEXT NOT NULL", False, False),
    ("street", "TEXT", False, False),
    ("x, y", "FLOAT (SVY21 coords)", False, False),
    ("market_segment", "TEXT (CCR/RCR/OCR)", False, False),
    ("area", "FLOAT (sqm)", False, False),
    ("floor_range", "TEXT (e.g. 06-10)", False, False),
    ("contract_date", "TEXT (MMYY)", False, False),
    ("type_of_sale", "TEXT (1=New/2=Sub/3=Resale)", False, False),
    ("price", "NUMERIC(15,2)", False, False),
    ("property_type", "TEXT", False, False),
    ("district", "TEXT (01-28)", False, False),
    ("tenure", "TEXT", False, False),
    ("nett_price", "NUMERIC(15,2)", False, False),
    ("ingested_at", "TIMESTAMP DEFAULT NOW()", False, False),
]
x1l, y1b, x1r, y1t = draw_table(ax, 0.8, 14.2, "ura_transactions", t1, width=10.0)

# ec_features (bottom-left)
t2 = [
    ("id", "UUID", True, False),
    ("transaction_id", "UUID → ura_transactions", False, True),
    ("project", "TEXT NOT NULL", False, False),
    ("district", "TEXT", False, False),
    ("area_sqm", "FLOAT", False, False),
    ("floor_mid", "INT (midpoint)", False, False),
    ("lease_commence_year", "INT", False, False),
    ("years_from_launch", "INT", False, False),
    ("price_psm", "FLOAT (price per sqm)", False, False),
    ("price", "NUMERIC(15,2)", False, False),
    ("created_at", "TIMESTAMP DEFAULT NOW()", False, False),
]
x2l, y2b, x2r, y2t = draw_table(ax, 0.8, 6.5, "ec_features", t2, width=10.0)

# model_registry (top-right)
t3 = [
    ("id", "UUID", True, False),
    ("model_name", "TEXT NOT NULL", False, False),
    ("model_version", "TEXT NOT NULL", False, False),
    ("algorithm", "TEXT", False, False),
    ("metrics", "JSONB", False, False),
    ("parameters", "JSONB", False, False),
    ("artifact_path", "TEXT", False, False),
    ("is_active", "BOOLEAN DEFAULT FALSE", False, False),
    ("created_at", "TIMESTAMP DEFAULT NOW()", False, False),
]
x3l, y3b, x3r, y3t = draw_table(ax, 13.2, 14.2, "model_registry", t3, width=10.0)

# prediction_logs (bottom-right)
t4 = [
    ("id", "UUID", True, False),
    ("model_version", "TEXT NOT NULL", False, False),
    ("input_features", "JSONB NOT NULL", False, False),
    ("predicted_price", "NUMERIC(15,2)", False, False),
    ("latency_ms", "FLOAT", False, False),
    ("created_at", "TIMESTAMP DEFAULT NOW()", False, False),
]
x4l, y4b, x4r, y4t = draw_table(ax, 13.2, 8.5, "prediction_logs", t4, width=10.0)

# ── RELATIONSHIP ARROWS ──

# ura_transactions → ec_features (vertical, left side)
mid_x1 = 5.8
ax.annotate(
    "", xy=(mid_x1, y2t + 0.05), xytext=(mid_x1, y1b - 0.05),
    arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.5, shrinkA=0, shrinkB=0)
)
ax.text(mid_x1 + 0.3, (y1b + y2t) / 2, "1 : 1", fontsize=11, color="#2563eb",
        fontweight="bold", va="center",
        bbox=dict(facecolor="white", edgecolor="#2563eb", boxstyle="round,pad=0.3", linewidth=1.5))

# model_registry → prediction_logs (vertical, right side)
mid_x2 = 18.2
ax.annotate(
    "", xy=(mid_x2, y4t + 0.05), xytext=(mid_x2, y3b - 0.05),
    arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.5, shrinkA=0, shrinkB=0)
)
ax.text(mid_x2 + 0.3, (y3b + y4t) / 2, "1 : N", fontsize=11, color="#2563eb",
        fontweight="bold", va="center",
        bbox=dict(facecolor="white", edgecolor="#2563eb", boxstyle="round,pad=0.3", linewidth=1.5))

# ── DATA FLOW ARROWS (horizontal, dashed) ──

flow_y_top = 11.0
ax.annotate(
    "", xy=(x3l - 0.1, flow_y_top), xytext=(x1r + 0.1, flow_y_top),
    arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.5, linestyle="--", shrinkA=0, shrinkB=0)
)
ax.text((x1r + x3l) / 2, flow_y_top + 0.3, "Feature Engineering + Training",
        ha="center", fontsize=11, color="#059669", fontweight="bold",
        bbox=dict(facecolor="#f0fdf4", edgecolor="#059669", boxstyle="round,pad=0.3", linewidth=1))

flow_y_bot = 5.5
ax.annotate(
    "", xy=(x4l - 0.1, flow_y_bot), xytext=(x2r + 0.1, flow_y_bot),
    arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.5, linestyle="--", shrinkA=0, shrinkB=0)
)
ax.text((x2r + x4l) / 2, flow_y_bot + 0.3, "Model Inference → Prediction Logging",
        ha="center", fontsize=11, color="#059669", fontweight="bold",
        bbox=dict(facecolor="#f0fdf4", edgecolor="#059669", boxstyle="round,pad=0.3", linewidth=1))

# ── LEGEND ──
ly = 0.6
ax.add_patch(patches.FancyBboxPatch((1.5, ly - 0.2), 0.55, 0.35,
             boxstyle="round,pad=0.04", facecolor="#fef2f2", edgecolor="#ef4444", linewidth=1))
ax.text(1.775, ly - 0.02, "PK", ha="center", va="center", fontsize=9, fontweight="bold", color="#dc2626")
ax.text(2.3, ly - 0.02, "Primary Key", fontsize=11, va="center", color="#334155")

ax.add_patch(patches.FancyBboxPatch((5.0, ly - 0.2), 0.55, 0.35,
             boxstyle="round,pad=0.04", facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1))
ax.text(5.275, ly - 0.02, "FK", ha="center", va="center", fontsize=9, fontweight="bold", color="#2563eb")
ax.text(5.8, ly - 0.02, "Foreign Key", fontsize=11, va="center", color="#334155")

ax.plot([8.5, 9.8], [ly, ly], color="#2563eb", lw=2.5)
ax.annotate("", xy=(9.8, ly), xytext=(9.5, ly),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.5))
ax.text(10.1, ly, "Relationship", fontsize=11, va="center", color="#334155")

ax.plot([13.5, 14.8], [ly, ly], "--", color="#059669", lw=2.5)
ax.annotate("", xy=(14.8, ly), xytext=(14.5, ly),
            arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.5))
ax.text(15.1, ly, "Data Flow", fontsize=11, va="center", color="#334155")

plt.savefig("docs/data_model.pdf", dpi=200, bbox_inches="tight", facecolor="#fafafa")
plt.savefig("docs/data_model.png", dpi=200, bbox_inches="tight", facecolor="#fafafa")
print("Saved: docs/data_model.pdf, docs/data_model.png")
