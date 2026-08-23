#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_ensemble.py - Standard ensemble vs. MC Dropout ensemble
(the "Ours w/o MC Dropout" ablation in the paper).

Example:
    python compare_ensemble.py --data_dir processed_data --model_dir checkpoints/
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import argparse
import glob

from fusion_models import get_model
from train import load_processed_data, UltrasoundDataset, preload_data

def calculate_metrics(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, auc, f1

def main():
    parser = argparse.ArgumentParser(description="Compare the Standard Ensemble and the MC Dropout Ensemble")
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help="Root directory containing FOLD1..FOLD5 and TEST splits")
    parser.add_argument('--model_dir', type=str, required=True,
                        help="Directory containing fold1_with_temp.pth .. fold5_with_temp.pth")
    parser.add_argument('--mc_iters', type=int, default=50, help="Number of MC Dropout samples")
    parser.add_argument('--allow_partial', action='store_true',
                        help="Allow running with fewer than 5 fold checkpoints")
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    processed_dir = args.data_dir

    if not os.path.isdir(processed_dir):
        print(f"Error: data directory not found: {processed_dir}")
        return

    _, test_pairs, classes = load_processed_data(processed_dir)

    if len(test_pairs) == 0:
        print("Error: no test samples were loaded; please check the data directory layout.")
        return
    preload_data(processed_dir, test_pairs)

    test_dataset = UltrasoundDataset(test_pairs, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)

    models_and_temps = []
    for i in range(1, 6):
        model_path = os.path.join(args.model_dir, f'fold{i}_with_temp.pth')
        if not os.path.exists(model_path):
            matches = glob.glob(os.path.join(args.model_dir, f'*fold{i}*.pth'))
            if not matches:
                print(f"Warning: checkpoint for fold {i} not found, skipped.")
                continue
            model_path = sorted(matches)[0]

        # pretrained=False: the checkpoint covers every parameter anyway
        model = get_model('prototype_enhanced', num_classes=len(classes), pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            temp = checkpoint.get('temperature', 1.0)
        else:
            model.load_state_dict(checkpoint)
            temp = 1.0

        model.to(device)
        models_and_temps.append((model, temp))
    if len(models_and_temps) == 0:
        print("Error: no fold checkpoints were found.")
        return
    if len(models_and_temps) < 5 and not args.allow_partial:
        print(f"Error: expected 5 fold checkpoints but found {len(models_and_temps)}. "
              f"Pass --allow_partial to run a partial ensemble anyway.")
        return
    print(f"Loaded {len(models_and_temps)} models, starting the comparison...")

    all_labels = []
    all_probs_std = []
    all_probs_mc = []

    with torch.no_grad():
        for bmode, doppler, labels in tqdm(test_loader, desc="Inference Progress"):
            bmode, doppler = bmode.to(device), doppler.to(device)

            batch_probs_std_models = []
            batch_probs_mc_models = []

            for model, temp in models_and_temps:
                # Mode A: standard ensemble (Dropout off)
                model.eval()

                outputs_dict = model.forward_with_details(bmode, doppler)
                logits_std = outputs_dict['logits'] / temp
                probs_std = F.softmax(logits_std, dim=1).cpu().numpy()
                batch_probs_std_models.append(probs_std)

                # Mode B: MC Dropout ensemble (only Dropout layers back in train mode)
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()

                mc_probs = []
                for _ in range(args.mc_iters):
                    outputs_dict = model.forward_with_details(bmode, doppler)
                    logits_mc = outputs_dict['logits'] / temp
                    mc_probs.append(F.softmax(logits_mc, dim=1).cpu().numpy())

                batch_probs_mc_models.append(np.mean(mc_probs, axis=0))

            # Soft voting over the 5 fold models
            all_probs_std.append(np.mean(batch_probs_std_models, axis=0))
            all_probs_mc.append(np.mean(batch_probs_mc_models, axis=0))
            all_labels.append(labels.numpy())

    probs_std = np.concatenate(all_probs_std, axis=0)
    preds_std = np.argmax(probs_std, axis=1)

    probs_mc = np.concatenate(all_probs_mc, axis=0)
    preds_mc = np.argmax(probs_mc, axis=1)

    labels = np.concatenate(all_labels, axis=0)

    acc_std, auc_std, f1_std = calculate_metrics(labels, probs_std, preds_std)
    acc_mc, auc_mc, f1_mc = calculate_metrics(labels, probs_mc, preds_mc)

    print(f"\n{'='*60}")
    print(f"{'Comparison Results (Test Set)':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} | {'Standard (no DO)':<20} | {'MC Dropout':<20}")
    print(f"{'-'*60}")
    print(f"{'Accuracy (%)':<15} | {acc_std:>18.2f} | {acc_mc:>18.2f}")
    print(f"{'Macro AUC':<15} | {auc_std:>18.4f} | {auc_mc:>18.4f}")
    print(f"{'Macro F1':<15} | {f1_std:>18.4f} | {f1_mc:>18.4f}")
    print(f"{'='*60}")

    diff = acc_mc - acc_std
    if diff > 0:
        print(f"\nConclusion: MC Dropout improves the base accuracy by {diff:.2f}%. Besides uncertainty estimation, it acts as an extra regularizing smoother.")
    elif diff < 0:
        print(f"\nConclusion: MC Dropout slightly lowers the accuracy by {abs(diff):.2f}%. This is expected: MC Dropout mainly measures uncertainty, and the injected noise can perturb a few borderline confident samples.")
    else:
        print(f"\nConclusion: both modes reach the same accuracy. MC Dropout preserves the base performance while additionally providing uncertainty information.")

if __name__ == "__main__":
    main()
