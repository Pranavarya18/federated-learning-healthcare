import os
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works in VS Code without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


RESULTS_DIR = "./results"


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Per-dataset convergence plots
# ─────────────────────────────────────────────

def plot_loss(losses, dataset="mnist", save=True):
    """Plot training loss across FL rounds."""
    _ensure_dir()
    rounds = list(range(1, len(losses) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, losses, marker='o', color='crimson', linewidth=2, markersize=7)
    ax.set_title(f"FL Training Loss — {dataset.upper()}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"{dataset}_loss.png")
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Saved: {path}")
    plt.close(fig)


def plot_accuracy(metrics, dataset="mnist", metric_label="Accuracy (%)", save=True):
    """Plot accuracy (or RMSE) across FL rounds."""
    _ensure_dir()
    rounds = list(range(1, len(metrics) + 1))

    color = 'steelblue' if 'Accuracy' in metric_label else 'darkorange'

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rounds, metrics, marker='s', color=color, linewidth=2, markersize=7)
    ax.set_title(f"FL {metric_label} — {dataset.upper()}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(metric_label)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        label_safe = metric_label.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
        path = os.path.join(RESULTS_DIR, f"{dataset}_{label_safe}.png")
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Saved: {path}")
    plt.close(fig)


def plot_convergence(losses, metrics, dataset="mnist", metric_label="Accuracy (%)", save=True):
    """Dual-axis convergence plot: loss + accuracy/RMSE on same figure."""
    _ensure_dir()
    rounds = list(range(1, len(losses) + 1))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(rounds, losses, marker='o', color='crimson', linewidth=2, label='Loss')
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Loss", color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    color2 = 'steelblue' if 'Accuracy' in metric_label else 'darkorange'
    ax2.plot(rounds, metrics, marker='s', color=color2, linewidth=2,
             linestyle='--', label=metric_label)
    ax2.set_ylabel(metric_label, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle(f"FL Convergence — {dataset.upper()}", fontsize=13, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax1.grid(True, alpha=0.2)
    fig.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, f"{dataset}_convergence.png")
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────
#  Communication Overhead Plot
# ─────────────────────────────────────────────

def plot_communication_overhead(overhead_mb_per_dataset: dict, save=True):
    """Bar chart showing cumulative communication overhead per dataset.

    Args:
        overhead_mb_per_dataset: e.g. {'MNIST': 12.3, 'Heart': 1.2, 'Diabetes': 0.9}
    """
    _ensure_dir()
    labels = list(overhead_mb_per_dataset.keys())
    values = list(overhead_mb_per_dataset.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=['steelblue', 'crimson', 'seagreen'], alpha=0.85)
    ax.set_title("Communication Overhead per Dataset", fontsize=13, fontweight='bold')
    ax.set_ylabel("Total Data Transferred (MB)")
    ax.set_xlabel("Dataset")
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f} MB", ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "communication_overhead.png")
        fig.savefig(path, dpi=150)
        print(f"[PLOT] Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────
#  Final Comparison Table Plot
# ─────────────────────────────────────────────

def plot_comparison_table(results: dict, save=True):
    """Render a summary comparison table as an image.

    Args:
        results: dict with keys = dataset names, values = dict of metrics
        Example:
            {
              'MNIST':    {'Final Accuracy (%)': 95.2, 'Final Loss': 0.18, 'RMSE': '-', 'Rounds': 5},
              'Heart':    {'Final Accuracy (%)': 82.0, 'Final Loss': 0.41, 'RMSE': '-', 'Rounds': 5},
              'Diabetes': {'Final Accuracy (%)': '-',  'Final Loss': 0.09, 'RMSE': 52.3, 'Rounds': 5},
            }
    """
    _ensure_dir()

    datasets = list(results.keys())
    if not datasets:
        return

    # Build table rows
    all_keys = list(results[datasets[0]].keys())
    col_labels = ['Dataset'] + all_keys
    rows = []
    for ds in datasets:
        row = [ds] + [str(results[ds].get(k, '-')) for k in all_keys]
        rows.append(row)

    fig, ax = plt.subplots(figsize=(10, 2 + len(rows) * 0.6))
    ax.axis('off')

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row shading
    for i, _ in enumerate(rows):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i + 1, j].set_facecolor('#ecf0f1')

    ax.set_title("Federated Learning — Final Results Comparison",
                 fontsize=13, fontweight='bold', pad=20)
    fig.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, "comparison_table.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {path}")
    plt.close(fig)
