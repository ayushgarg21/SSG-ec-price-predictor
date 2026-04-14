"""Generate clean data model diagram — no overlaps."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(28, 20))
ax.set_xlim(0, 28)
ax.set_ylim(0, 20)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(14, 19.3, "EC Price Predictor — Data Model",
        ha="center", fontsize=26, fontweight="bold", color="#0f172a")
ax.text(14, 18.8, "PostgreSQL 16  |  4 Tables  |  UUID Primary Keys  |  JSONB Metadata",
        ha="center", fontsize=14, color="#64748b")


def draw_table(ax, x, y, title, columns, width=11.0):
    row_h = 0.48
    header_h = 0.65
    pad = 0.15
    total_h = header_h + len(columns) * row_h + pad * 2

    # Shadow
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.12, y - total_h + 0.12), width - 0.04, total_h - 0.04,
        boxstyle="round,pad=0.12", facecolor="#e2e8f0", edgecolor="none", zorder=0
    ))
    # Body
    ax.add_patch(patches.FancyBboxPatch(
        (x, y - total_h), width, total_h,
        boxstyle="round,pad=0.12", facecolor="white", edgecolor="#94a3b8", linewidth=1.5, zorder=1
    ))
    # Header
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.04, y - header_h + 0.04), width - 0.08, header_h - 0.04,
        boxstyle="round,pad=0.08", facecolor="#1e40af", edgecolor="none", zorder=2
    ))
    ax.add_patch(patches.Rectangle(
        (x + 0.04, y - header_h + 0.04), width - 0.08, 0.2,
        facecolor="#1e40af", edgecolor="none", zorder=2
    ))
    ax.text(x + width / 2, y - header_h / 2 + 0.02, title,
            ha="center", va="center", fontsize=15, fontweight="bold", color="white",
            fontfamily="monospace", zorder=3)

    for i, (name, dtype, is_pk, is_fk) in enumerate(columns):
        cy = y - header_h - pad - (i + 0.5) * row_h

        if i % 2 == 0:
            ax.add_patch(patches.Rectangle(
                (x + 0.12, cy - row_h / 2 + 0.03), width - 0.24, row_h - 0.06,
                facecolor="#f8fafc", edgecolor="none", zorder=1.5
            ))

        name_x = x + 0.4
        if is_pk:
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.25, cy - 0.15), 0.6, 0.3,
                boxstyle="round,pad=0.04", facecolor="#fef2f2", edgecolor="#ef4444", linewidth=1, zorder=3
            ))
            ax.text(x + 0.55, cy, "PK", ha="center", va="center", fontsize=9, fontweight="bold", color="#dc2626", zorder=4)
            name_x = x + 1.05
        elif is_fk:
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.25, cy - 0.15), 0.6, 0.3,
                boxstyle="round,pad=0.04", facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1, zorder=3
            ))
            ax.text(x + 0.55, cy, "FK", ha="center", va="center", fontsize=9, fontweight="bold", color="#2563eb", zorder=4)
            name_x = x + 1.05

        ax.text(name_x, cy, name, va="center", fontsize=12,
                color="#0f172a", fontweight="bold" if (is_pk or is_fk) else "normal", zorder=3)
        ax.text(x + width - 0.35, cy, dtype, va="center", ha="right",
                fontsize=10.5, color="#94a3b8", zorder=3)

    return x, y - total_h, x + width, y


# ── TABLES (well-spaced 2x2 grid) ──

# Top-left: ura_transactions
t1 = [
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
    ("district", "TEXT (01-28)", False, False),
    ("tenure", "TEXT", False, False),
    ("nett_price", "NUMERIC(15,2)", False, False),
    ("ingested_at", "TIMESTAMP", False, False),
]
x1l, y1b, x1r, y1t = draw_table(ax, 0.8, 18.0, "ura_transactions", t1, width=12.0)

# Top-right: model_registry
t3 = [
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
x3l, y3b, x3r, y3t = draw_table(ax, 15.2, 18.0, "model_registry", t3, width=12.0)

# Bottom-left: ec_features
t2 = [
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
x2l, y2b, x2r, y2t = draw_table(ax, 0.8, 8.2, "ec_features", t2, width=12.0)

# Bottom-right: prediction_logs
t4 = [
    ("id", "UUID", True, False),
    ("model_version", "TEXT NOT NULL", False, False),
    ("input_features", "JSONB NOT NULL", False, False),
    ("predicted_price", "NUMERIC(15,2)", False, False),
    ("latency_ms", "FLOAT", False, False),
    ("created_at", "TIMESTAMP", False, False),
]
x4l, y4b, x4r, y4t = draw_table(ax, 15.2, 8.2, "prediction_logs", t4, width=12.0)

# ── RELATIONSHIP ARROWS ──

# ura_transactions → ec_features (left column, vertical)
mid_x1 = 6.8
gap1_top = y1b - 0.15
gap1_bot = y2t + 0.15
ax.annotate("", xy=(mid_x1, gap1_bot), xytext=(mid_x1, gap1_top),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=3, shrinkA=0, shrinkB=0))
ax.text(mid_x1 + 0.4, (gap1_top + gap1_bot) / 2, "1 : 1",
        fontsize=13, color="#2563eb", fontweight="bold", va="center",
        bbox=dict(facecolor="white", edgecolor="#2563eb", boxstyle="round,pad=0.35", linewidth=1.5))

# model_registry → prediction_logs (right column, vertical)
mid_x2 = 21.2
gap2_top = y3b - 0.15
gap2_bot = y4t + 0.15
ax.annotate("", xy=(mid_x2, gap2_bot), xytext=(mid_x2, gap2_top),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=3, shrinkA=0, shrinkB=0))
ax.text(mid_x2 + 0.4, (gap2_top + gap2_bot) / 2, "1 : N",
        fontsize=13, color="#2563eb", fontweight="bold", va="center",
        bbox=dict(facecolor="white", edgecolor="#2563eb", boxstyle="round,pad=0.35", linewidth=1.5))

# ── DATA FLOW (horizontal dashed) ──

flow_y1 = 13.5
ax.annotate("", xy=(x3l - 0.2, flow_y1), xytext=(x1r + 0.2, flow_y1),
            arrowprops=dict(arrowstyle="-|>", color="#059669", lw=3, linestyle="--", shrinkA=0, shrinkB=0))
ax.text((x1r + x3l) / 2, flow_y1 + 0.4, "Feature Engineering + Training",
        ha="center", fontsize=13, color="#059669", fontweight="bold",
        bbox=dict(facecolor="#f0fdf4", edgecolor="#059669", boxstyle="round,pad=0.35", linewidth=1.5))

flow_y2 = 5.5
ax.annotate("", xy=(x4l - 0.2, flow_y2), xytext=(x2r + 0.2, flow_y2),
            arrowprops=dict(arrowstyle="-|>", color="#059669", lw=3, linestyle="--", shrinkA=0, shrinkB=0))
ax.text((x2r + x4l) / 2, flow_y2 + 0.4, "Model Inference → Prediction Logging",
        ha="center", fontsize=13, color="#059669", fontweight="bold",
        bbox=dict(facecolor="#f0fdf4", edgecolor="#059669", boxstyle="round,pad=0.35", linewidth=1.5))

# ── LEGEND ──
ly = 0.8
items = [
    (1.5, "PK", "#fef2f2", "#ef4444", "#dc2626", "Primary Key"),
    (5.5, "FK", "#eff6ff", "#3b82f6", "#2563eb", "Foreign Key"),
]
for lx, label, bg, border, tc, desc in items:
    ax.add_patch(patches.FancyBboxPatch((lx, ly - 0.2), 0.6, 0.4,
                 boxstyle="round,pad=0.05", facecolor=bg, edgecolor=border, linewidth=1))
    ax.text(lx + 0.3, ly, label, ha="center", va="center", fontsize=10, fontweight="bold", color=tc)
    ax.text(lx + 0.85, ly, desc, fontsize=12, va="center", color="#334155")

ax.plot([10, 11.5], [ly, ly], color="#2563eb", lw=3)
ax.annotate("", xy=(11.5, ly), xytext=(11.2, ly),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=3))
ax.text(11.8, ly, "Relationship", fontsize=12, va="center", color="#334155")

ax.plot([16, 17.5], [ly, ly], "--", color="#059669", lw=3)
ax.annotate("", xy=(17.5, ly), xytext=(17.2, ly),
            arrowprops=dict(arrowstyle="-|>", color="#059669", lw=3))
ax.text(17.8, ly, "Data Flow", fontsize=12, va="center", color="#334155")

plt.savefig("docs/data_model.pdf", dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig("docs/data_model.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: docs/data_model.pdf, docs/data_model.png")
