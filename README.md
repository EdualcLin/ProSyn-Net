# ProSyn-Net

Official implementation of **"ProSyn-Net: A Reliable Prototypical Synergy Network for Imbalanced Dermatologic Multimodal Ultrasound Diagnosis"** (MICCAI 2026, poster).

ProSyn-Net is a multimodal diagnostic framework for paired B-mode / color Doppler dermatologic ultrasound, designed around three synergistic modules:

1. **Dynamic Cross-Attention Fusion** – dual ConvNeXt-Tiny branches (first two stages frozen) extract global features `f_b, f_d ∈ R^D`; each modality's feature is refined by a symmetric cross-attention block that uses the other modality's feature as key/value (Eq. 1 in the paper), and a lightweight dynamic weight generator produces sample-adaptive fusion weights
   `f_fused = α_b · Attention(f_b, f_d) + α_d · Attention(f_d, f_b)`.
2. **Non-linear Prototypical Mapping** – learnable geometric prototypes (one per class) define a squared-Euclidean distance profile `d(x)`, which is decoded by a two-layer MLP (hidden dim 512, BatchNorm + ReLU + Dropout 0.3) instead of a rigid linear Voronoi classifier. Trained with `L_Focal + λ_c·L_Compact + λ_s·L_Sep` (λ_c = 0.2, λ_s = 0.02).
3. **Uncertainty-Aware Manifold Ensemble** – soft voting over the 5 cross-validation folds with temperature scaling and M = 50 MC Dropout stochastic passes; predictive uncertainty `U(x) = 1 − max_k p̄_k(x)` supports selective deferral of ambiguous cases.

## Results (in-house imbalanced 6-class dataset)

| Metric | Value |
|---|---|
| Accuracy | 80.14% |
| Macro AUC | 96.66% |
| Macro F1 | 78.60% |
| Macro Sensitivity | 76.82% |
| Macro Specificity | 95.47% |

Single-fold mean F1 is 69.32%; the 5-fold manifold ensemble yields a +9.28% absolute gain. Disabling MC Dropout gives 76.37% F1 (−2.23%).

## Repository structure

```
├── fusion_models.py          # ProSyn-Net model (cross-attention fusion + prototype classifier)
├── focal_loss.py             # Focal Loss + prototype compactness/separation losses
├── train.py                  # 5-fold CV training + temperature calibration + ensemble test
├── evaluate_ensemble.py      # Reproduce the ensemble test metrics from trained fold checkpoints
├── compare_ensemble.py       # Standard ensemble vs. MC Dropout ensemble (ablation)
├── plot_rejection_curve.py   # Accuracy-Rejection (clinical deferral) curves
├── smoke_test.py             # Minimal forward/loss/backward sanity check (no data needed)
├── requirements.txt
└── requirements-lock.txt     # Pinned reference environment
```

## Installation

```bash
conda create -n prosyn python=3.10 -y
conda activate prosyn
pip install -r requirements.txt
# or, for the pinned reference environment: pip install -r requirements-lock.txt
```

Tested environment: Python 3.12, PyTorch 2.9.0 + CUDA 12.8, torchvision 0.24.0, numpy 2.3.5, scikit-learn 1.7.1, scipy 1.16.1, tqdm 4.67.1, Pillow 10.4.0, matplotlib 3.10.8 (4 × NVIDIA RTX 5880 Ada). Older PyTorch versions (>= 2.0) should work as well.

## Data preparation

The code expects paired B-mode / color Doppler images arranged as follows:

```
processed_data/
├── FOLD1/
│   └── <CLASS_NAME>/          # e.g. AK, BCC, Bowen, Paget, SCC, SK
│       ├── case001.jpg        # B-mode image
│       └── case001c.jpg       # paired color Doppler image ("c" suffix; "case001 c.jpg" also works)
├── FOLD2/ ... FOLD5/
└── TEST/                      # independent hold-out test set
```

- Class names are discovered automatically from the (alphabetically sorted) subdirectory names.
- Training uses the 5 folds for cross-validation; `TEST/` is only used for final evaluation.
- The in-house dataset is private patient data and is **not** released. Due to institutional/data-governance restrictions, checkpoints trained on this cohort are not distributed either; run `train.py` to obtain comparable models.

## Training

```bash
python train.py --data_dir processed_data --output_dir results
```

Paper hyper-parameters are the defaults: 80 epochs, AdamW (weight decay 5e-3), 5-epoch linear warmup + MultiStepLR (milestones 20/40, γ 0.5), gradient clipping 1.0, Focal Loss (γ = 2, label smoothing 0.05, inverse-frequency class weights), λ_c = 0.2, λ_s = 0.02. Batch size 16 and lr 1e-4 are per-GPU values: multiple GPUs are used automatically via DDP, with the learning rate scaled linearly by the world size (e.g. 4 GPUs → global batch 64, lr 4e-4).

Training saves `best_prosyn_net_fold{1..5}(_with_temp).pth` and runs the ensemble test at the end.

## Evaluation (reproducing the paper numbers)

With the five fold checkpoints in a directory (e.g. `checkpoints/fold1_with_temp.pth` … `fold5_with_temp.pth`):

```bash
# Ensemble metrics (Accuracy / AUC / Macro F1) + uncertainty npz export
python evaluate_ensemble.py --data_dir processed_data --model_dir checkpoints/

# Ablation: standard ensemble vs. MC Dropout ensemble
python compare_ensemble.py --data_dir processed_data --model_dir checkpoints/

# Accuracy-Rejection curve (Fig. 3 of the paper)
python plot_rejection_curve.py --pattern "ensemble_results_*.npz" --output fig3.png
```

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{lin2026prosynnet,
  title={ProSyn-Net: A Reliable Prototypical Synergy Network for Imbalanced Dermatologic Multimodal Ultrasound Diagnosis},
  author={Lin, Jicheng and Dai, Xiangning and Wang, Beidi and Feng, Juncai and Liu, Haotian and Yu, Chenke and Liu, Ruimeng and Qin, Ziwei and Zhao, Yujing and Luo, Ye},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work was supported by the General Program of the National Natural Science Foundation of China under Grant 62276189, the Grants of Tongji University Medicine-X Interdisciplinary Research Initiative under Grant Nos. 20250554-YB-09, 2025-0650-YB-17, and 2026-0674-YB-05, and the Fundamental Research Funds for the Central Universities under Grant 2025-1-ZD-02.
