# coding: utf-8
"""Part B analysis: scaling laws (3.1) and architectural comparison (3.2).

Produces two figures:
  Figure 1 — Loss vs Non-Embedding Parameters (log-log, linear fit).
             All architectures (baseline + variants) on the same graph.
  Figure 2 — Loss vs Token Position (context length effect).
             From the best-performing baseline model.

Usage:
    python analyze_partb.py --results_dir ../results/part_b
"""

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Part B scaling law analysis')
parser.add_argument('--results_dir', type=str, default='../results/part_b',
                    help='directory containing Part B results JSON files')
parser.add_argument('--part_a_results', type=str, default='../results',
                    help='directory containing Part A baseline results (optional)')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, args.results_dir)
PART_A_DIR = os.path.join(SCRIPT_DIR, args.part_a_results)


# ─── Color & marker scheme ────────────────────────────────────────────────────

ARCH_STYLES = {
    'baseline':                 {'color': '#1f77b4', 'marker': 'o'},
    'qk_norm':                  {'color': '#ff7f0e', 'marker': 's'},
    'attn_gate':                {'color': '#2ca02c', 'marker': '^'},
    'value_emb':                {'color': '#d62728', 'marker': 'D'},
    'qk_norm+attn_gate':       {'color': '#9467bd', 'marker': 'v'},
    'qk_norm+value_emb':       {'color': '#8c564b', 'marker': 'p'},
    'attn_gate+value_emb':      {'color': '#17becf', 'marker': 'P'},
    'qk_norm+attn_gate+value_emb': {'color': '#e377c2', 'marker': '*'},
}


def load_results(json_dir):
    """Load all results_*.json files from a directory."""
    records = []
    if not os.path.isdir(json_dir):
        return records
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith('.json'):
            continue
        if fname.startswith('results_'):
            with open(os.path.join(json_dir, fname)) as f:
                r = json.load(f)
                r['_file'] = fname
                r['_dir'] = json_dir
                records.append(r)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Loss vs Non-Embedding Parameters (log-log, linear fit)
#           baseline (§3.1) vs enhanced architecture (§3.2)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scaling_laws(records):
    """All architectures on one log-log plot with per-arch linear fits."""
    arch_data = {}
    for r in records:
        arch = r['config'].get('arch', 'baseline')
        non_emb = r['config'].get('non_embedding_params')
        if non_emb is None:
            continue  # skip old Part A results that lack this field
        best_epoch = r['best_epoch']
        best_loss = r['valid_epoch_loss'][best_epoch - 1]
        arch_data.setdefault(arch, []).append((non_emb, best_loss))

    fig, ax = plt.subplots(figsize=(12, 8))

    for arch, points in sorted(arch_data.items()):
        points = sorted(points, key=lambda x: x[0])
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        style = ARCH_STYLES.get(arch, {'color': '#333333', 'marker': 'x'})
        color = style['color']
        marker = style['marker']

        ax.scatter(xs, ys, color=color, marker=marker, s=80, zorder=5,
                   label=arch, edgecolors='black', linewidth=0.5)

        if len(xs) >= 2:
            log_x = np.log10(xs)
            log_y = np.log10(ys)
            slope, intercept = np.polyfit(log_x, log_y, 1)

            x_fit = np.logspace(log_x.min(), log_x.max(), 50)
            y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
            ax.plot(x_fit, y_fit, '--', color=color, alpha=0.6, linewidth=1.5)

            mid_x = float(np.sqrt(x_fit[0] * x_fit[-1]))
            mid_y = float(np.sqrt(y_fit[0] * y_fit[-1]))
            ax.annotate(f'$\\alpha={slope:.3f}$', (mid_x, mid_y),
                        color=color, fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Non-Embedding Parameters', fontsize=13)
    ax.set_ylabel('Best Validation Loss', fontsize=13)
    ax.set_title('Scaling Law: Validation Loss vs Model Size\n(Part B §3.1 + §3.2)', fontsize=14)
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    for fmt in ('svg', 'png'):
        path = os.path.join(RESULTS_DIR, f'partb_fig1_scaling_laws.{fmt}')
        plt.savefig(path, dpi=150)
        print(f"Figure 1 saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Loss vs Token Position (context length effect, §3.1 Study 2)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_position_loss(record):
    """Average loss vs token position group from a single well-trained model."""
    pos_groups = record.get('position_groups', [])
    if not pos_groups:
        print("No position_groups in record — train.py may need re-run.")
        return

    labels = [g['range'] for g in pos_groups]
    losses = [g['loss'] for g in pos_groups]
    positions = []
    for label in labels:
        lo, hi = label.split('-')
        positions.append((int(lo) + int(hi)) / 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions, losses, 'b-o', markersize=6, linewidth=1.5)
    ax.set_xlabel('Token Position', fontsize=13)
    ax.set_ylabel('Average Loss', fontsize=13)
    ax.set_title(
        f'Loss vs Token Position (Context Length Effect)\n'
        f'{record["_file"]}  |  arch={record["config"]["arch"]}  '
        f'non-emb={record["config"]["non_embedding_params"]:,}',
        fontsize=12
    )
    ax.grid(True, alpha=0.3)

    for i in [0, len(positions) // 4, len(positions) // 2, -1]:
        ax.annotate(labels[i], (positions[i], losses[i]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.1', fc='lightyellow', alpha=0.8))

    plt.tight_layout()
    for fmt in ('svg', 'png'):
        path = os.path.join(RESULTS_DIR, f'partb_fig2_position_loss.{fmt}')
        plt.savefig(path, dpi=150)
        print(f"Figure 2 saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load Part B results
    records = load_results(RESULTS_DIR)

    # Also try to load Part A baseline for comparison (if exists)
    if os.path.isdir(PART_A_DIR) and PART_A_DIR != RESULTS_DIR:
        part_a_records = load_results(PART_A_DIR)
        # Only take baseline arch from Part A
        for r in part_a_records:
            if r['config'].get('arch', 'baseline') == 'baseline':
                r['_file'] = f"[Part A] {r['_file']}"
                records.append(r)

    if not records:
        print(f"No results_*.json files found in {RESULTS_DIR}")
        print("Run scripts/sweep_3.1_scaling.sh first.")
        return

    print(f"Loaded {len(records)} result files")

    # ── Figure 1: Scaling law + architectural comparison (§3.1 Study 1 + §3.2) ──
    plot_scaling_laws(records)

    # ── Figure 2: Position-dependent loss (§3.1 Study 2) ──
    # Only consider records that actually have position_groups data
    baseline_records = [r for r in records
                        if r['config'].get('arch', 'baseline') == 'baseline'
                        and r.get('position_groups')]
    if baseline_records:
        best = min(baseline_records, key=lambda r: r['best_valid_ppl'])
        print(f"\nPosition-loss source: {best['_file']} (PPL={best['best_valid_ppl']:.2f})")
        plot_position_loss(best)
    else:
        print("No baseline results — cannot produce position-loss plot.")


if __name__ == '__main__':
    main()
