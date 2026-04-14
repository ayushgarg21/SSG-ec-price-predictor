"""Generate clean system architecture diagram."""

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
ax.text(12, 15.5, "EC Price Predictor — System Architecture",
        ha="center", fontsize=24, fontweight="bold", color="#0f172a")
ax.text(12, 15.0, "Docker Compose  |  PostgreSQL 16  |  FastAPI  |  Streamlit  |  XGBoost / LightGBM / CatBoost",
        ha="center", fontsize=12, color="#64748b")


def draw_component(ax, x, y, w, h, title, items, color="#1e40af", title_size=13):
    """Draw a component box. x,y = bottom-left."""
    # Shadow
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.08, y - 0.08), w, h,
        boxstyle="round,pad=0.15", facecolor="#cbd5e1", edgecolor="none", zorder=0
    ))
    # Body
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, linewidth=2.5, zorder=1
    ))
    # Header
    hdr_h = 0.55
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.03, y + h - hdr_h), w - 0.06, hdr_h,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="none", zorder=2
    ))
    ax.add_patch(patches.Rectangle(
        (x + 0.03, y + h - hdr_h), w - 0.06, 0.15,
        facecolor=color, edgecolor="none", zorder=2
    ))
    ax.text(x + w / 2, y + h - hdr_h / 2, title,
            ha="center", va="center", fontsize=title_size, fontweight="bold", color="white", zorder=3)

    for i, item in enumerate(items):
        iy = y + h - hdr_h - 0.35 - i * 0.38
        ax.text(x + 0.35, iy, f"•  {item}", fontsize=10, color="#334155", va="center", zorder=3)

    return x, y, x + w, y + h


def draw_arrow(ax, x1, y1, x2, y2, label="", color="#475569", dashed=False):
    style = "--" if dashed else "-"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, linestyle=style, shrinkA=3, shrinkB=3)
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        offset = 0.3 if x1 != x2 else 0.0
        ax.text(mx, my + offset, label, ha="center", va="center", fontsize=10, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.25", linewidth=1.2, alpha=0.95),
                zorder=5)


def draw_layer_label(ax, y, label, color):
    ax.add_patch(patches.FancyBboxPatch(
        (0.3, y - 0.25), 3.0, 0.5,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="none", alpha=0.12, zorder=0
    ))
    ax.text(1.8, y, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=color, zorder=1)


# ═══════════════════════════════════════════
# LAYER 1: DATA INGESTION (y: 11-14)
# ═══════════════════════════════════════════
draw_layer_label(ax, 12.8, "DATA INGESTION", "#059669")

draw_component(ax, 3.8, 11.2, 5.0, 2.5, "URA API", [
    "Private residential transactions",
    "4 batches by postal district",
    "Auth: AccessKey + daily Token",
    "Updated every Tue & Fri",
], color="#059669")

draw_component(ax, 10.0, 11.2, 5.0, 2.5, "Ingestion Pipeline", [
    "curl-based (bypasses WAF)",
    "Exponential backoff retry",
    "Circuit breaker (5 failures)",
    "Flatten nested JSON → rows",
], color="#059669")

draw_component(ax, 16.5, 11.2, 5.5, 2.5, "PostgreSQL 16", [
    "139,268 transactions loaded",
    "4 tables (see Data Model)",
    "UUID PKs, JSONB metadata",
    "Docker volume persistence",
], color="#059669")

draw_arrow(ax, 8.8, 12.45, 10.0, 12.45, "curl + retry", "#059669")
draw_arrow(ax, 15.0, 12.45, 16.5, 12.45, "bulk INSERT", "#059669")

# ═══════════════════════════════════════════
# LAYER 2: ML PIPELINE (y: 6.5-10)
# ═══════════════════════════════════════════
draw_layer_label(ax, 8.8, "ML PIPELINE", "#7c3aed")

draw_component(ax, 3.8, 6.8, 5.0, 2.8, "Feature Engineering", [
    "24 leakage-free features",
    "Lag-1 year district/project stats",
    "Appreciation ratio = PSM / launch",
    "Target encoding (train-only KFold)",
    "Temporal split FIRST, then features",
], color="#7c3aed")

draw_component(ax, 10.0, 6.8, 5.0, 2.8, "Model Training", [
    "Optuna HPO (60 trials x 3 algos)",
    "XGBoost / LightGBM / CatBoost",
    "Weighted ensemble optimization",
    "Temporal train → val → test",
    "Quantile regression for intervals",
], color="#7c3aed")

draw_component(ax, 16.5, 6.8, 5.5, 2.8, "Evaluation", [
    "R² = 0.88  |  MAPE = 5.4%",
    "SHAP TreeExplainer",
    "Segmented eval (district, age)",
    "Residual analysis",
    "Experiment tracking (JSONL)",
], color="#7c3aed")

draw_arrow(ax, 8.8, 8.2, 10.0, 8.2, "features", "#7c3aed")
draw_arrow(ax, 15.0, 8.2, 16.5, 8.2, "metrics", "#7c3aed")

# Postgres → Feature Engineering (vertical)
draw_arrow(ax, 6.3, 11.2, 6.3, 9.6, "SQL query", "#64748b", dashed=True)

# ═══════════════════════════════════════════
# LAYER 3: SERVING (y: 1.5-5.5)
# ═══════════════════════════════════════════
draw_layer_label(ax, 4.5, "SERVING LAYER", "#2563eb")

draw_component(ax, 3.8, 2.2, 5.0, 3.0, "FastAPI", [
    "POST /predict",
    "POST /predict/milestones",
    "GET /health",
    "SHAP explanations per request",
    "Prediction intervals (80% PI)",
    "Rate limit (60/min) + TTL cache",
], color="#2563eb")

draw_component(ax, 10.0, 2.2, 5.0, 3.0, "Model Serving", [
    "model.joblib (LightGBM)",
    "serving_lookups.joblib",
    "Prediction → Postgres logging",
    "Non-root Docker container",
    "2 uvicorn workers",
    "Healthcheck every 30s",
], color="#2563eb")

draw_component(ax, 16.5, 2.2, 5.5, 3.0, "Streamlit Frontend", [
    "Milestone comparison (5yr vs 10yr)",
    "Custom prediction mode",
    "Prediction interval display",
    "SHAP feature contributions",
    "EC project selector",
    "Real-time API status",
], color="#e65100")

draw_arrow(ax, 8.8, 3.7, 10.0, 3.7, "load model", "#2563eb")
draw_arrow(ax, 15.0, 3.7, 16.5, 3.7, "HTTP JSON", "#e65100")

# Model artifacts → Serving (vertical)
draw_arrow(ax, 12.5, 6.8, 12.5, 5.2, "model.joblib", "#64748b", dashed=True)

# ── DEPLOYMENT BAR ──
ax.add_patch(patches.FancyBboxPatch(
    (3.8, 0.4), 18.2, 1.0,
    boxstyle="round,pad=0.15", facecolor="#f0fdf4", edgecolor="#059669", linewidth=2, zorder=2
))
ax.text(12.9, 0.9, "LIVE DEPLOYMENT",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#059669")
ax.text(12.9, 0.55, "API: http://172.234.215.236:8000/docs    |    Frontend: http://172.234.215.236:8501    |    GitHub: github.com/ayushgarg21/SSG-ec-price-predictor",
        ha="center", va="center", fontsize=10, color="#059669")

plt.savefig("docs/architecture_diagram.pdf", dpi=200, bbox_inches="tight", facecolor="#fafafa")
plt.savefig("docs/architecture_diagram.png", dpi=200, bbox_inches="tight", facecolor="#fafafa")
print("Saved: docs/architecture_diagram.pdf, docs/architecture_diagram.png")
