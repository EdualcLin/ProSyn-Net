#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_test.py - Minimal end-to-end sanity check (no data or GPU required).

Verifies model forward shapes, the detailed forward path, the combined loss
backward (including gradient flow to the prototypes), and a checkpoint
save/load roundtrip. Run: python smoke_test.py
"""

import torch
from fusion_models import get_model
from focal_loss import FocalLoss, PrototypeNetLoss


def main():
    torch.manual_seed(42)
    num_classes = 6
    model = get_model('prototype_enhanced', num_classes=num_classes, pretrained=False)

    n_params = sum(p.numel() for p in model.parameters())
    assert abs(n_params / 1e6 - 61.0) < 0.5, f"unexpected parameter count: {n_params/1e6:.2f}M"
    print(f"[1/4] parameter count: {n_params/1e6:.1f}M")

    bmode = torch.randn(2, 3, 224, 224)
    doppler = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 5])

    logits = model(bmode, doppler)
    assert logits.shape == (2, num_classes)
    details = model(bmode, doppler, return_details=True)
    assert details['fused_features'].shape == (2, 768)
    assert details['modal_weights'].shape == (2, 2)
    print("[2/4] forward shapes OK (plain + return_details)")

    criterion = PrototypeNetLoss(FocalLoss(gamma=2.0, label_smoothing=0.05),
                                 compact_weight=0.2, separation_weight=0.02)
    loss, loss_details = criterion(details['logits'], details['fused_features'],
                                   model.get_prototypes(), labels)
    model.zero_grad()
    loss.backward()
    pgrad = model.prototype_classifier.prototypes.grad
    assert pgrad is not None and pgrad.abs().sum() > 0, "no gradient reaching the prototypes"
    print(f"[3/4] loss backward OK (total={loss.item():.4f}, prototypes receive gradients)")

    state = model.state_dict()
    model2 = get_model('prototype_enhanced', num_classes=num_classes, pretrained=False)
    model2.load_state_dict(state, strict=True)
    print("[4/4] checkpoint strict save/load roundtrip OK")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
