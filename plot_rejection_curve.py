#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_rejection_curve.py - Accuracy-Rejection curves of the three uncertainty
metrics from one or more ensemble result npz files (Fig. 3 of the paper).

Usage:
    python plot_rejection_curve.py                      # uses ./ensemble_results_*.npz
    python plot_rejection_curve.py --pattern "runs/ensemble_results_*.npz" --output fig3.png
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

def calculate_curve(y_true, y_pred, uncertainties, num_points=51):
    sorted_indices = np.argsort(uncertainties)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    total_samples = len(y_true)
    rejection_rates = np.linspace(0, 0.5, num_points)
    accuracies = []

    for rate in rejection_rates:
        reject_count = int(total_samples * rate)
        if reject_count == total_samples:
            break
        remain_true = y_true_sorted[reject_count:]
        remain_pred = y_pred_sorted[reject_count:]
        acc = np.mean(remain_true == remain_pred) * 100
        accuracies.append(acc)

    return rejection_rates * 100, np.array(accuracies)

def main():
    parser = argparse.ArgumentParser(description="Plot Accuracy-Rejection curves from ensemble result npz files")
    parser.add_argument('--pattern', type=str, default='ensemble_results_*.npz',
                        help="Glob pattern of the npz files to aggregate")
    parser.add_argument('--output', type=str, default='miccai_metrics_comparison.png',
                        help="Output figure filename")
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))

    if len(files) == 0:
        print(f"Error: no files matched the pattern '{args.pattern}'.")
        return

    print(f"Found {len(files)} result file(s), aggregating...")

    all_acc_mc = []
    all_acc_ent = []
    all_acc_1max = []
    rates = None
    base_acc_list = []

    for f in files:
        data = np.load(f)
        y_true = data['y_true']
        y_pred = data['y_pred']

        base_acc_list.append(np.mean(y_true == y_pred) * 100)

        rates, acc_mc = calculate_curve(y_true, y_pred, data['uncert_mc'])
        _, acc_ent = calculate_curve(y_true, y_pred, data['uncert_entropy'])
        _, acc_1max = calculate_curve(y_true, y_pred, data['uncert_1minusmax'])

        all_acc_mc.append(acc_mc)
        all_acc_ent.append(acc_ent)
        all_acc_1max.append(acc_1max)

    mean_mc, std_mc = np.mean(all_acc_mc, axis=0), np.std(all_acc_mc, axis=0)
    mean_ent, std_ent = np.mean(all_acc_ent, axis=0), np.std(all_acc_ent, axis=0)
    mean_1max, std_1max = np.mean(all_acc_1max, axis=0), np.std(all_acc_1max, axis=0)

    avg_base_acc = np.mean(base_acc_list)

    plt.figure(figsize=(10, 6.5))

    plt.plot(rates, mean_ent, marker='s', markersize=5, linestyle='-', color='#d62728', linewidth=2, label='Softmax Entropy')
    plt.fill_between(rates, mean_ent - std_ent, mean_ent + std_ent, color='#d62728', alpha=0.15)

    plt.plot(rates, mean_1max, marker='^', markersize=5, linestyle='-', color='#2ca02c', linewidth=2, label='1 - Max Probability (Ours)')
    plt.fill_between(rates, mean_1max - std_1max, mean_1max + std_1max, color='#2ca02c', alpha=0.15)

    plt.plot(rates, mean_mc, marker='o', markersize=5, linestyle='-', color='#1f77b4', linewidth=2, label='MC Dropout Variance')
    plt.fill_between(rates, mean_mc - std_mc, mean_mc + std_mc, color='#1f77b4', alpha=0.15)

    plt.axhline(y=avg_base_acc, color='gray', linestyle='--', linewidth=2, label=f'Base Ensemble Accuracy ({avg_base_acc:.1f}%)')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Comparison of Uncertainty Metrics (Averaged over {len(files)} Ensemble Runs)', fontsize=14, fontweight='bold')
    plt.xlabel('Rejection Rate (%) - Top Uncertain Samples Referred to Experts', fontsize=12)
    plt.ylabel('Accuracy on Remaining Samples (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Figure saved to: {args.output}")

if __name__ == "__main__":
    main()
