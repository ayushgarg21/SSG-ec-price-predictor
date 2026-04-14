"""Generate clean system architecture diagram — no overlaps."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(28, 20))
ax.set_xlim(0, 28)
ax.set_ylim(0, 20)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(14, 19.4, "EC Price Predictor — System Architecture",
        ha="center", fontsize=26, fontweight="bold", color="#0f172a")
ax.text(14, 18.9, "Docker Compose  |  PostgreSQL 16  |  FastAPI  |  Streamlit  |  XGBoost / LightGBM / CatBoost",
        ha="center", fontsize=13, color="#64748b")


def box(ax, x, y, w, h, title, items, color="#1e40af"):
    """x,y = bottom-left. Returns (left, bottom, right, top)."""
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.1, y - 0.1), w, h,
        boxstyle="round,pad=0.15", facecolor="#e2e8f0", edgecolor="none", zorder=0))
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, linewidth=2.5, zorder=1))

    hdr = 0.6
    ax.add_patch(patches.FancyBboxPatch(
        (x + 0.04, y + h - hdr), w - 0.08, hdr,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="none", zorder=2))
    ax.add_patch(patches.Rectangle(
        (x + 0.04, y + h - hdr), w - 0.08, 0.18,
        facecolor=color, edgecolor="none", zorder=2))
    ax.text(x + w/2, y + h - hdr/2, title,
            ha="center", va="center", fontsize=14, fontweight="bold", color="white", zorder=3)

    for i, item in enumerate(items):
        iy = y + h - hdr - 0.45 - i * 0.42
        ax.text(x + 0.4, iy, f"•  {item}", fontsize=11, color="#334155", va="center", zorder=3)

    return x, y, x + w, y + h


def arrow(ax, x1, y1, x2, y2, label="", color="#475569", dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5,
                                linestyle="--" if dashed else "-", shrinkA=4, shrinkB=4))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        offset = 0.35 if abs(x2-x1) > abs(y2-y1) else 0.0
        side_offset = 0.0 if abs(x2-x1) > abs(y2-y1) else 0.7
        ax.text(mx + side_offset, my + offset, label, ha="center", va="center",
                fontsize=10, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.25",
                          linewidth=1.2, alpha=0.95), zorder=5)


def layer_label(ax, y, text, color):
    ax.add_patch(patches.FancyBboxPatch(
        (0.3, y - 0.3), 3.2, 0.6,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="none", alpha=0.12))
    ax.text(1.9, y, text, ha="center", va="center",
            fontsize=11, fontweight="bold", color=color)


# ═══════════════════════════════════════
# LAYER 1: DATA INGESTION (y: 14.5 - 18)
# ═══════════════════════════════════════
layer_label(ax, 16.8, "DATA INGESTION", "#059669")

L, B, R, T = box(ax, 4.0, 14.8, 6.0, 3.0, "URA API", [
    "Private residential transactions",
    "4 batches by postal district",
    "Auth: AccessKey + daily Token",
    "Updated every Tue & Fri",
    "5 years of historical data",
], color="#059669")

L2, B2, R2, T2 = box(ax, 11.0, 14.8, 6.0, 3.0, "Ingestion Pipeline", [
    "curl-based (bypasses WAF)",
    "Exponential backoff retry",
    "Circuit breaker (5 failures)",
    "Flatten nested JSON → rows",
    "Data validation checks",
], color="#059669")

L3, B3, R3, T3 = box(ax, 18.5, 14.8, 6.5, 3.0, "PostgreSQL 16", [
    "139,268 transactions loaded",
    "4 tables (see Data Model)",
    "UUID PKs, JSONB metadata",
    "Docker volume persistence",
    "Prediction audit logging",
], color="#059669")

arrow(ax, R, 16.3, L2, 16.3, "curl + retry", "#059669")
arrow(ax, R2, 16.3, L3, 16.3, "bulk INSERT", "#059669")

# ═══════════════════════════════════════
# LAYER 2: ML PIPELINE (y: 8.5 - 13)
# ═══════════════════════════════════════
layer_label(ax, 11.3, "ML PIPELINE", "#7c3aed")

box(ax, 4.0, 9.0, 6.0, 3.5, "Feature Engineering", [
    "24 leakage-free features",
    "Lag-1 year district/project stats",
    "Full market (139k) for district",
    "Appreciation ratio target",
    "Target encoding (train-only KFold)",
    "Temporal split FIRST",
], color="#7c3aed")

box(ax, 11.0, 9.0, 6.0, 3.5, "Model Training", [
    "Optuna HPO (60 trials × 3 algos)",
    "XGBoost / LightGBM / CatBoost",
    "Weighted ensemble optimisation",
    "Temporal train → val → test",
    "Quantile regression for intervals",
    "Zero look-ahead bias",
], color="#7c3aed")

box(ax, 18.5, 9.0, 6.5, 3.5, "Evaluation & Artifacts", [
    "R² = 0.88  |  MAPE = 5.4%",
    "SHAP TreeExplainer",
    "68% prediction interval coverage",
    "Segmented eval (district, age)",
    "Residual analysis plots",
    "Experiment tracking (JSONL)",
], color="#7c3aed")

arrow(ax, 10.0, 10.75, 11.0, 10.75, "features", "#7c3aed")
arrow(ax, 17.0, 10.75, 18.5, 10.75, "metrics", "#7c3aed")
arrow(ax, 7.0, 14.8, 7.0, 12.5, "SQL query", "#64748b", dashed=True)

# ═══════════════════════════════════════
# LAYER 3: SERVING (y: 2.5 - 7.5)
# ═══════════════════════════════════════
layer_label(ax, 5.8, "SERVING LAYER", "#2563eb")

box(ax, 4.0, 3.0, 6.0, 3.8, "FastAPI (Docker)", [
    "POST /predict",
    "POST /predict/milestones",
    "GET /health",
    "SHAP explanations per request",
    "Prediction intervals (80% PI)",
    "Rate limiting (60 req/min)",
    "TTL prediction cache",
], color="#2563eb")

box(ax, 11.0, 3.0, 6.0, 3.8, "Model Serving", [
    "model.joblib (LightGBM)",
    "serving_lookups.joblib",
    "StandardScaler preprocessing",
    "Prediction → Postgres logging",
    "Non-root Docker container",
    "2 uvicorn workers",
    "Healthcheck every 30s",
], color="#2563eb")

box(ax, 18.5, 3.0, 6.5, 3.8, "Streamlit Frontend", [
    "Milestone comparison (5yr vs 10yr)",
    "Custom prediction mode",
    "Prediction interval display",
    "SHAP feature contributions",
    "EC project selector dropdown",
    "Real-time API health status",
    "Deployed at :8501",
], color="#e65100")

arrow(ax, 10.0, 4.9, 11.0, 4.9, "load model", "#2563eb")
arrow(ax, 17.0, 4.9, 18.5, 4.9, "HTTP JSON", "#e65100")
arrow(ax, 14.0, 9.0, 14.0, 6.8, "model.joblib", "#64748b", dashed=True)

# ── DEPLOYMENT BAR ──
ax.add_patch(patches.FancyBboxPatch(
    (4.0, 0.6), 21.0, 1.2,
    boxstyle="round,pad=0.2", facecolor="#f0fdf4", edgecolor="#059669", linewidth=2, zorder=2))
ax.text(14.5, 1.25, "LIVE DEPLOYMENT", ha="center", fontsize=14, fontweight="bold", color="#059669")
ax.text(14.5, 0.8,
        "API: http://172.234.215.236:8000/docs    |    Frontend: http://172.234.215.236:8501    |    GitHub: github.com/ayushgarg21/SSG-ec-price-predictor",
        ha="center", fontsize=11, color="#059669")

plt.savefig("docs/architecture_diagram.pdf", dpi=150, bbox_inches="tight", facecolor="white")
plt.savefig("docs/architecture_diagram.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: docs/architecture_diagram.pdf, docs/architecture_diagram.png")
