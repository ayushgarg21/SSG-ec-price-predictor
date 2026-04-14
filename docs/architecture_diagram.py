"""Generate system architecture diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(11, 13.5, "EC Price Predictor — System Architecture",
        ha="center", fontsize=22, fontweight="bold", color="#0f172a")
ax.text(11, 13.05, "Docker Compose  |  PostgreSQL 16  |  FastAPI  |  Streamlit",
        ha="center", fontsize=12, color="#64748b")

def draw_box(ax, x, y, w, h, title, items, color="#1e40af", icon=""):
    """Draw a rounded component box."""
    box = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor="white", edgecolor=color, linewidth=2.5, zorder=2
    )
    ax.add_patch(box)
    # Header bar
    ax.add_patch(patches.FancyBboxPatch(
        (x, y + h - 0.6), w, 0.6, boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=color, linewidth=2, zorder=3
    ))
    ax.add_patch(patches.Rectangle((x, y + h - 0.6), w, 0.2, facecolor=color, edgecolor="none", zorder=3))

    ax.text(x + w/2, y + h - 0.3, f"{icon}  {title}" if icon else title,
            ha="center", va="center", fontsize=12, fontweight="bold", color="white", zorder=4)

    for i, item in enumerate(items):
        ax.text(x + 0.3, y + h - 1.0 - i * 0.35, f"• {item}",
                fontsize=9.5, color="#334155", va="center", zorder=3)

def draw_arrow(ax, x1, y1, x2, y2, label="", color="#334155", style="-|>"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=2.5, connectionstyle="arc3,rad=0")
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.25, label, ha="center", fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))

# ── Layer labels ──
for ly, label, color in [(12.2, "DATA INGESTION", "#059669"), (8.0, "ML PIPELINE", "#7c3aed"), (3.8, "SERVING LAYER", "#2563eb")]:
    ax.add_patch(patches.FancyBboxPatch(
        (0.3, ly), 2.5, 0.5, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="none", alpha=0.15, zorder=0
    ))
    ax.text(1.55, ly + 0.25, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=color, zorder=1)

# ═══════════════════════════════════════════
# DATA INGESTION LAYER
# ═══════════════════════════════════════════

draw_box(ax, 3.2, 10.8, 4.0, 2.0, "URA API", [
    "Private residential transactions",
    "4 batches (by postal district)",
    "Authentication: AccessKey + Token",
], color="#059669")

draw_box(ax, 8.5, 10.8, 4.0, 2.0, "Ingestion Pipeline", [
    "curl-based client (bypasses WAF)",
    "Retry + circuit breaker",
    "Flatten nested JSON → rows",
], color="#059669")

draw_box(ax, 14.5, 10.8, 5.5, 2.0, "PostgreSQL (Docker)", [
    "139,268 transactions",
    "4 tables: transactions, features,",
    "  model_registry, prediction_logs",
], color="#059669")

draw_arrow(ax, 7.2, 11.8, 8.5, 11.8, "curl + retry", "#059669")
draw_arrow(ax, 12.5, 11.8, 14.5, 11.8, "bulk INSERT", "#059669")

# ═══════════════════════════════════════════
# ML PIPELINE LAYER
# ═══════════════════════════════════════════

draw_box(ax, 3.2, 6.5, 4.5, 2.5, "Feature Engineering", [
    "24 leakage-free features",
    "Lag-1 year district/project stats",
    "Appreciation ratio target",
    "Target encoding (train-only)",
], color="#7c3aed")

draw_box(ax, 9.0, 6.5, 4.5, 2.5, "Model Training", [
    "Optuna HPO (60 trials × 3 algos)",
    "XGBoost / LightGBM / CatBoost",
    "Weighted ensemble",
    "Temporal train/val/test split",
], color="#7c3aed")

draw_box(ax, 15.0, 6.5, 5.0, 2.5, "Evaluation & Artifacts", [
    "R²=0.88, MAPE=5.4%",
    "SHAP explanations",
    "Prediction intervals (quantile reg)",
    "Experiment tracking (JSONL)",
], color="#7c3aed")

draw_arrow(ax, 7.7, 7.75, 9.0, 7.75, "features", "#7c3aed")
draw_arrow(ax, 13.5, 7.75, 15.0, 7.75, "metrics", "#7c3aed")
draw_arrow(ax, 17.2, 10.8, 17.2, 9.0, "SQL query", "#64748b")

# ═══════════════════════════════════════════
# SERVING LAYER
# ═══════════════════════════════════════════

draw_box(ax, 3.2, 1.8, 5.0, 2.8, "FastAPI (Docker)", [
    "POST /predict — single prediction",
    "POST /predict/milestones — 5yr vs 10yr",
    "GET /health — model status",
    "SHAP + prediction intervals",
    "Rate limiting + TTL cache",
], color="#2563eb")

draw_box(ax, 9.5, 1.8, 4.5, 2.8, "Serving Infrastructure", [
    "Model bundle (joblib)",
    "Serving lookups (district/project)",
    "Prediction logging → Postgres",
    "Non-root Docker container",
    "2 uvicorn workers",
], color="#2563eb")

draw_box(ax, 15.5, 1.8, 5.0, 2.8, "Streamlit Frontend", [
    "Milestone comparison tab",
    "Custom prediction tab",
    "Prediction intervals display",
    "SHAP feature contributions",
    "Project selector dropdown",
], color="#e65100")

draw_arrow(ax, 8.2, 3.2, 9.5, 3.2, "load model", "#2563eb")
draw_arrow(ax, 14.0, 3.2, 15.5, 3.2, "HTTP API", "#e65100")

# Vertical flow: artifacts → serving
draw_arrow(ax, 5.7, 6.5, 5.7, 4.6, "model.joblib", "#64748b")

# ── Deployment info ──
ax.add_patch(patches.FancyBboxPatch(
    (3.2, 0.3), 17.3, 0.9, boxstyle="round,pad=0.15",
    facecolor="#f0fdf4", edgecolor="#059669", linewidth=1.5, zorder=2
))
ax.text(11.85, 0.75, "Deployed at:   API → http://172.234.215.236:8000/docs   |   Frontend → http://172.234.215.236:8501   |   GitHub → github.com/ayushgarg21/SSG-ec-price-predictor",
        ha="center", va="center", fontsize=10, color="#059669", fontweight="bold", zorder=3)

plt.tight_layout(pad=0.3)
plt.savefig("docs/architecture_diagram.pdf", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("docs/architecture_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved: docs/architecture_diagram.pdf, docs/architecture_diagram.png")
