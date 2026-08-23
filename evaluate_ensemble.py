#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate_ensemble.py - Reproduce the ProSyn-Net 5-fold ensemble test results.

Example:
    python evaluate_ensemble.py --data_dir processed_data --model_dir checkpoints/
"""

import os
import glob
import argparse
import torch

from train import load_processed_data, preload_data, UltrasoundDataset, run_ensemble_test


def main():
    parser = argparse.ArgumentParser(description="ProSyn-Net 5-fold ensemble evaluation")
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help="Root directory containing FOLD1..FOLD5 and TEST splits")
    parser.add_argument('--model_dir', type=str, required=True,
                        help="Directory containing fold1_with_temp.pth .. fold5_with_temp.pth")
    parser.add_argument('--mc_iterations', type=int, default=50,
                        help="Number of MC Dropout stochastic forward passes")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--allow_partial', action='store_true',
                        help="Allow running with fewer than 5 fold checkpoints")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not os.path.isdir(args.data_dir):
        print(f"Error: data directory not found: {args.data_dir}")
        return

    _, test_pairs, classes = load_processed_data(args.data_dir)
    if len(test_pairs) == 0:
        print("Error: no test samples were loaded; please check the data directory layout.")
        return
    print(f"Classes: {classes}")
    print(f"Test samples: {len(test_pairs)}")

    preload_data(args.data_dir, test_pairs)

    test_dataset = UltrasoundDataset(test_pairs, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True
    )

    model_paths_and_temps = []
    for i in range(1, 6):
        model_path = os.path.join(args.model_dir, f'fold{i}_with_temp.pth')
        if not os.path.exists(model_path):
            matches = glob.glob(os.path.join(args.model_dir, f'*fold{i}*.pth'))
            if not matches:
                print(f"Warning: checkpoint for fold {i} not found in {args.model_dir}, skipped.")
                continue
            model_path = sorted(matches)[0]
        # The temperature stored inside the checkpoint takes precedence
        model_paths_and_temps.append((model_path, 1.0))

    if len(model_paths_and_temps) == 0:
        print("Error: no fold checkpoints were found.")
        return
    if len(model_paths_and_temps) < 5 and not args.allow_partial:
        print(f"Error: expected 5 fold checkpoints but found {len(model_paths_and_temps)}. "
              f"Pass --allow_partial to run a partial ensemble anyway.")
        return

    run_ensemble_test(model_paths_and_temps, test_loader, device, classes,
                      mc_iterations=args.mc_iterations)


if __name__ == "__main__":
    main()
