#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - ProSyn-Net training script (MICCAI 2026)

5-fold CV training + temperature calibration + ensemble test with MC Dropout.
Data layout and usage: see README.md.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from PIL import Image
import warnings
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import _LRScheduler
import random
from scipy import stats
from datetime import datetime
import shutil
import argparse
import sys

script_dir_import = os.path.dirname(os.path.realpath(__file__))
if script_dir_import not in sys.path:
    sys.path.append(script_dir_import)
from fusion_models import get_model
from focal_loss import FocalLoss, PrototypeNetLoss

os.environ['PYTHONHASHSEED'] = str(42)

def seed_worker(worker_id):
    """Sets the seed for a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

warnings.filterwarnings('ignore')

g_preloaded_data = {}

def find_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class WarmupLR(_LRScheduler):
    """Linear-warmup scheduler wrapping a main scheduler."""
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr, after_scheduler):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.initial_lr * (1 - alpha) + self.target_lr * alpha]
        return self.after_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch < self.warmup_epochs:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            self.after_scheduler.step(epoch - self.warmup_epochs)

class UltrasoundDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.preloaded_data = g_preloaded_data
        self.augment = augment
        self._placeholder_img = Image.new('RGB', (224, 224), (0, 0, 0))
        self._warned_keys = set()
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        bmode_key = pair['bmode']
        doppler_key = pair['doppler']

        bmode_img = self.preloaded_data.get(bmode_key)
        if bmode_img is None:
            if bmode_key not in self._warned_keys:
                print(f"\n[Warning] B-mode image key not found in preloaded data: '{bmode_key}'. Using a placeholder.")
                self._warned_keys.add(bmode_key)
            bmode_img = self._placeholder_img

        doppler_img = self.preloaded_data.get(doppler_key)
        if doppler_img is None:
            if doppler_key not in self._warned_keys:
                print(f"\n[Warning] Doppler image key not found in preloaded data: '{doppler_key}'. Using a placeholder.")
                self._warned_keys.add(doppler_key)
            doppler_img = self._placeholder_img

        transform = self.aug_transform if self.augment else self.base_transform
        if self.augment:
            # Paired-augmentation sync: one shared random parameter set for both
            # modalities, so paired B-mode/Doppler images stay registered
            sync_seed = torch.randint(0, 2**31 - 1, (1,)).item()
            torch.manual_seed(sync_seed)
            bmode_t = transform(bmode_img)
            torch.manual_seed(sync_seed)
            doppler_t = transform(doppler_img)
            return bmode_t, doppler_t, pair['label']
        return transform(bmode_img), transform(doppler_img), pair['label']


def calculate_sensitivity_specificity(cm, classes):
    """Per-class sensitivity and specificity from a confusion matrix."""
    n_classes = len(classes)
    sensitivity_dict = {}
    specificity_dict = {}

    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sensitivity_dict[classes[i]] = sensitivity

        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - tp
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_dict[classes[i]] = specificity

    return sensitivity_dict, specificity_dict


def print_sensitivity_specificity(sensitivity_dict, specificity_dict, classes, class_counts=None):
    print("\n" + "="*80)
    print("Per-class Sensitivity (Recall) and Specificity:")
    print("="*80)
    print(f"{'Class':<15} {'Sensitivity':<25} {'Specificity':<25}")
    print("-"*80)

    sensitivities = []
    specificities = []

    for cls in classes:
        sens = sensitivity_dict[cls]
        spec = specificity_dict[cls]
        sensitivities.append(sens)
        specificities.append(spec)
        print(f"{cls:<15} {sens:<25.4f} {spec:<25.4f}")

    print("-"*80)
    macro_sens = np.mean(sensitivities)
    macro_spec = np.mean(specificities)
    print(f"{'Macro avg':<15} {macro_sens:<25.4f} {macro_spec:<25.4f}")

    if class_counts is not None:
        total_samples = sum(class_counts)
        weighted_sens = sum(s * c for s, c in zip(sensitivities, class_counts)) / total_samples
        weighted_spec = sum(s * c for s, c in zip(specificities, class_counts)) / total_samples
        print(f"{'Weighted avg':<15} {weighted_sens:<25.4f} {weighted_spec:<25.4f}")

    print("="*80)


class ModelTrainer:
    def __init__(self, gpu, world_size):
        self.gpu = gpu
        self.world_size = world_size
        self.device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')
        self.is_ddp = world_size > 1

    def _get_unwrapped_model(self, model):
        return model.module if hasattr(model, 'module') else model

    def _forward_with_loss(self, model, bmode, doppler, labels, criterion):
        # Forward THROUGH the (possibly DDP-wrapped) model so the DDP reducer
        # tracks the iteration and synchronizes gradients
        outputs_dict = model(bmode, doppler, return_details=True)
        logits = outputs_dict['logits']

        features = outputs_dict.get('fused_features')
        if features is None:
            features = outputs_dict.get('features')
        if features is None:
            raise ValueError("outputs_dict must contain a 'fused_features' or 'features' key")

        # get_prototypes() returns the same Parameter objects used in the forward
        # pass, so their gradients flow through the DDP-managed parameters
        model_unwrapped = self._get_unwrapped_model(model)
        if hasattr(model_unwrapped, 'get_prototypes'):
            prototypes = model_unwrapped.get_prototypes()
        elif 'prototypes' in outputs_dict:
            prototypes = outputs_dict['prototypes']
        else:
            raise ValueError("A prototype model must provide get_prototypes() or include 'prototypes' in the output dict")

        loss, loss_details = criterion(logits, features, prototypes, labels)
        return logits, loss, loss_details

    def calibrate_temperature(self, model, val_loader):
        """Fit a temperature on the validation set (log-parameterized for stability)."""
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for bmode, doppler, labels in val_loader:
                bmode, doppler = bmode.to(self.device), doppler.to(self.device)
                model_unwrapped = self._get_unwrapped_model(model)
                outputs_dict = model_unwrapped.forward_with_details(bmode, doppler)
                outputs = outputs_dict['logits']

                all_logits.append(outputs)
                all_labels.append(labels.to(self.device))

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        log_temperature = torch.nn.Parameter(torch.zeros(1).to(self.device))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            loss = nn.CrossEntropyLoss()(all_logits / temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        final_temperature = torch.exp(log_temperature).item()

        if final_temperature < 0.1 or final_temperature > 10.0:
            if self.gpu == 0:
                print(f"  [Warning] Temperature out of safe range (T={final_temperature:.4f}); falling back to T=1.0")
            return 1.0

        if self.gpu == 0:
            print(f"  Temperature scaling calibrated: T = {final_temperature:.4f}")

        return final_temperature

    def train_model(self, model, train_loader, val_loader, train_sampler, epochs=80, lr=1e-4,
                   model_name="prosyn_net", focal_gamma=2.0,
                   prototype_compact_weight=0.2, prototype_separation_weight=0.02,
                   class_weights=None):

        model = model.to(self.device)
        weights = class_weights.to(self.device) if class_weights is not None else None

        cls_criterion = FocalLoss(alpha=weights, gamma=focal_gamma, label_smoothing=0.05)
        criterion = PrototypeNetLoss(
            cls_criterion=cls_criterion,
            compact_weight=prototype_compact_weight,
            separation_weight=prototype_separation_weight
        )
        if self.gpu == 0:
            print(f"Using Prototype Net Loss (gamma={focal_gamma}, compact={prototype_compact_weight}, separation={prototype_separation_weight})")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)

        main_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
        warmup_epochs = 5
        scheduler = WarmupLR(optimizer, warmup_epochs=warmup_epochs, initial_lr=lr/10, target_lr=lr, after_scheduler=main_scheduler)

        best_val_acc = 0
        loss_details_history = []

        for epoch in range(epochs):
            if train_sampler and self.is_ddp:
                train_sampler.set_epoch(epoch)
            model.train()
            train_correct, train_total = 0, 0
            pbar = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/{epochs} [Train]', disable=(self.gpu != 0))

            for bmode, doppler, labels in pbar:
                bmode, doppler, labels = bmode.to(self.device), doppler.to(self.device), labels.to(self.device)

                outputs, total_loss, loss_details = self._forward_with_loss(
                    model, bmode, doppler, labels, criterion
                )
                loss_details_history.append(loss_details)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'total_loss': f"{loss_details['total_loss']:.4f}",
                    'cls_loss': f"{loss_details['cls_loss']:.4f}",
                    'compact': f"{loss_details.get('compact_loss', 0):.4f}",
                    'separation': f"{loss_details.get('separation_loss', 0):.4f}"
                })

            train_acc = 100. * train_correct / train_total

            model.eval()
            val_correct, val_total = 0, 0
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for bmode, doppler, labels in val_loader:
                    bmode, doppler, labels = bmode.to(self.device), doppler.to(self.device), labels.to(self.device)
                    outputs, val_loss, _ = self._forward_with_loss(
                        model, bmode, doppler, labels, criterion
                    )

                    val_loss_sum += val_loss.item()
                    val_batches += 1

                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0
            scheduler.step()

            if self.gpu == 0:
                print(f'\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {avg_val_loss:.4f}')
                if len(loss_details_history) > 0:
                    recent_details = loss_details_history[-1]
                    print(f'  Loss details - total: {recent_details["total_loss"]:.4f}, '
                          f'cls: {recent_details["cls_loss"]:.4f}, '
                          f'compact: {recent_details.get("compact_loss", 0):.4f}, '
                          f'separation: {recent_details.get("separation_loss", 0):.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.gpu == 0:
                    if self.is_ddp:
                        state_dict_to_save = model.module.state_dict()
                    else:
                        state_dict_to_save = model.state_dict()
                    torch.save(state_dict_to_save, f'best_{model_name}_fold{getattr(self, "current_fold", 0)}.pth')
                    print(f'  Saved a new best model (accuracy: {val_acc:.2f}%)')

        if self.is_ddp:
            dist.barrier()

        best_model_path = f'best_{model_name}_fold{getattr(self, "current_fold", 0)}.pth'
        state_dict = torch.load(best_model_path, map_location=self.device)

        if self.is_ddp:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        if self.gpu == 0:
            print(f"\nRunning temperature scaling calibration...")
            temperature = self.calibrate_temperature(model, val_loader)
            save_dict = {
                'model_state_dict': state_dict,
                'temperature': temperature
            }
            torch.save(save_dict, f'best_{model_name}_fold{getattr(self, "current_fold", 0)}_with_temp.pth')
            print(f'  Model and temperature parameter saved')
        else:
            temperature = 1.0

        if self.is_ddp:
            dist.barrier()

        return best_val_acc, model, temperature

    def test_model(self, model, test_loader, classes, temperature=1.0, mc_iterations=50):
        model.to(self.device).eval()
        # MC Dropout: keep only Dropout layers in train mode
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        all_labels_list = []
        all_probs_list = []
        all_vars_list = []
        all_distances_list = []

        model_unwrapped = self._get_unwrapped_model(model)

        with torch.no_grad():
            for bmode, doppler, labels in tqdm(test_loader, desc="MC Dropout Testing", disable=(self.gpu != 0)):
                bmode, doppler = bmode.to(self.device), doppler.to(self.device)

                batch_probs_mc = []
                batch_distances_mc = []

                for _ in range(mc_iterations):
                    outputs_dict = model_unwrapped.forward_with_details(bmode, doppler)
                    outputs = outputs_dict['logits']
                    distances = outputs_dict.get('distances')

                    scaled_outputs = outputs / temperature
                    probs = F.softmax(scaled_outputs, dim=1)
                    batch_probs_mc.append(probs.cpu().numpy())
                    if distances is not None:
                        batch_distances_mc.append(distances.cpu().numpy())

                mean_batch_probs = np.mean(batch_probs_mc, axis=0)
                var_batch_probs = np.var(batch_probs_mc, axis=0)
                uncertainty = np.mean(var_batch_probs, axis=1)

                all_probs_list.append(mean_batch_probs)
                all_labels_list.append(labels.numpy())
                all_vars_list.append(uncertainty)

                if batch_distances_mc:
                    mean_distances = np.mean(batch_distances_mc, axis=0)
                    all_distances_list.append(mean_distances)

        all_probs = np.concatenate(all_probs_list, axis=0)
        all_labels = np.concatenate(all_labels_list, axis=0)
        all_uncertainty = np.concatenate(all_vars_list, axis=0)
        all_preds = np.argmax(all_probs, axis=1)

        acc = accuracy_score(all_labels, all_preds) * 100
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = 0.0
        f1 = f1_score(all_labels, all_preds, average='macro')

        cm = confusion_matrix(all_labels, all_preds)
        sensitivity_dict, specificity_dict = calculate_sensitivity_specificity(cm, classes)

        if self.gpu == 0:
            print("\n" + "="*80)
            print(f"Confusion matrix (Temperature={temperature:.4f}, MC Dropout):")
            print("="*80)
            print(cm)

            class_counts = [np.sum(all_labels == i) for i in range(len(classes))]
            print_sensitivity_specificity(sensitivity_dict, specificity_dict, classes, class_counts)

            print(f"\nClassification report (Temperature={temperature:.4f}, MC Dropout):")
            print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

        result = {
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'uncertainty': np.mean(all_uncertainty),
            'confusion_matrix': cm,
            'sensitivity': sensitivity_dict,
            'specificity': specificity_dict,
            'temperature': temperature
        }

        if all_distances_list:
            result['distances'] = np.concatenate(all_distances_list, axis=0)

        return result

def build_pairs_under(processed_dir, dest_group, classes):
    """Collect (B-mode, Doppler, label) pairs from one split directory.

    A Doppler image is recognized by the 'c' suffix ('namec.jpg' or 'name c.jpg')
    and matched to the B-mode image with the same base name.
    """
    pairs = []
    for i, name in enumerate(classes):
        class_dir = os.path.join(processed_dir, dest_group, name)
        if not os.path.exists(class_dir):
            continue

        files = os.listdir(class_dir)
        image_files = {f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))}

        for d_file in image_files:
            d_file_lower = d_file.lower()
            base_name, ext = "", ""

            if d_file_lower.endswith(' c.jpg') or d_file_lower.endswith(' c.png') or d_file_lower.endswith(' c.jpeg'):
                base_name = os.path.splitext(d_file)[0][:-2]
                ext = os.path.splitext(d_file)[1]
            elif d_file_lower.endswith('c.jpg') or d_file_lower.endswith('c.png') or d_file_lower.endswith('c.jpeg'):
                base_name = os.path.splitext(d_file)[0][:-1]
                ext = os.path.splitext(d_file)[1]
            else:
                continue

            b_file_name_lower = (base_name + ext).lower()
            actual_b_file = next((f for f in image_files if f.lower() == b_file_name_lower), None)

            if actual_b_file:
                pairs.append({
                    'bmode': os.path.join(dest_group, name, actual_b_file),
                    'doppler': os.path.join(dest_group, name, d_file),
                    'label': i
                })

    unique_pairs = []
    seen_b_files = set()
    for pair in pairs:
        b_lower = pair['bmode'].lower()
        if b_lower not in seen_b_files:
            unique_pairs.append(pair)
            seen_b_files.add(b_lower)

    return unique_pairs

def discover_classes(processed_dir):
    """Class names = sorted subdirectory names of the TEST (or FOLD1) split."""
    for split in ('TEST', 'FOLD1'):
        split_dir = os.path.join(processed_dir, split)
        if os.path.isdir(split_dir):
            classes = sorted(d for d in os.listdir(split_dir)
                             if os.path.isdir(os.path.join(split_dir, d)))
            if classes:
                return classes
    return []

def load_processed_data(processed_dir, classes=None):
    if classes is None:
        classes = discover_classes(processed_dir)
    fold_pairs = {}
    for k in range(1, 6):
        dest = f'FOLD{k}'
        fold_pairs[dest] = build_pairs_under(processed_dir, dest, classes)
    test_pairs = build_pairs_under(processed_dir, 'TEST', classes)
    return fold_pairs, test_pairs, classes

def preload_data(data_dir, pairs):
    global g_preloaded_data
    pbar = tqdm(pairs, desc="Preloading images into RAM")
    for pair in pbar:
        for key in ['bmode', 'doppler']:
            img_path_key = pair[key]
            if img_path_key not in g_preloaded_data:
                full_path = os.path.join(data_dir, img_path_key)
                try:
                    img = Image.open(full_path).convert('RGB')
                    g_preloaded_data[img_path_key] = img.copy()
                    img.close()
                except Exception as e:
                    raise RuntimeError(f"Failed to load image: {full_path}") from e

def run_ensemble_test(model_paths_and_temps, test_loader, device, classes, mc_iterations=50):
    """5-fold ensemble test with temperature scaling + MC Dropout; saves npz for the rejection curve."""
    models_and_temps = []
    for path, temp in model_paths_and_temps:
        # pretrained=False: the checkpoint covers every parameter anyway
        model = get_model('prototype_enhanced', num_classes=len(classes), pretrained=False)
        try:
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                temperature = checkpoint.get('temperature', temp)
            else:
                model.load_state_dict(checkpoint)
                temperature = temp
        except Exception:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            temperature = temp

        model.to(device)
        model.eval()
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        models_and_temps.append((model, temperature))

    print(f"\nLoaded {len(models_and_temps)} models, temperatures: {[f'{t:.4f}' for _, t in models_and_temps]}")

    all_labels_list = []
    all_samples_probs_from_models = []
    all_samples_distances_from_models = []
    all_samples_mcvar_from_models = []

    with torch.no_grad():
        for bmode, doppler, labels in tqdm(test_loader, desc="Ensemble Temperature-scaled MC Dropout Testing"):
            bmode, doppler = bmode.to(device), doppler.to(device)

            batch_mean_probs_from_models = []
            batch_mean_dists_from_models = []
            batch_mcvar_from_models = []

            for model, temperature in models_and_temps:
                batch_probs_mc = []
                batch_dists_mc = []

                for _ in range(mc_iterations):
                    outputs_dict = model.forward_with_details(bmode, doppler)
                    outputs = outputs_dict['logits']
                    distances = outputs_dict.get('distances')

                    scaled_outputs = outputs / temperature
                    probs = F.softmax(scaled_outputs, dim=1)
                    batch_probs_mc.append(probs.cpu().numpy())

                    if distances is not None:
                        dists_np = distances.cpu().numpy()  # [B, num_classes]
                        # distance margin (2nd nearest - nearest); negated: smaller margin = higher uncertainty
                        sorted_dists = np.sort(dists_np, axis=1)
                        margin = sorted_dists[:, 1] - sorted_dists[:, 0]
                        batch_dists_mc.append(-margin)

                mean_batch_probs = np.mean(batch_probs_mc, axis=0)
                batch_mean_probs_from_models.append(mean_batch_probs)

                # Within-model MC Dropout variance: variance across the M stochastic
                # passes of THIS model, averaged over classes -> [B]
                mcvar = np.mean(np.var(np.stack(batch_probs_mc, axis=0), axis=0), axis=1)
                batch_mcvar_from_models.append(mcvar)

                if batch_dists_mc:
                    mean_batch_dists = np.mean(batch_dists_mc, axis=0)
                    batch_mean_dists_from_models.append(mean_batch_dists)

            batch_mean_probs_from_models = np.array(batch_mean_probs_from_models).transpose(1, 0, 2)
            all_samples_probs_from_models.append(batch_mean_probs_from_models)
            all_labels_list.append(labels.numpy())

            batch_mcvar_from_models = np.array(batch_mcvar_from_models).transpose(1, 0)  # [B, n_models]
            all_samples_mcvar_from_models.append(batch_mcvar_from_models)

            if batch_mean_dists_from_models:
                batch_mean_dists_from_models = np.array(batch_mean_dists_from_models).transpose(1, 0)
                all_samples_distances_from_models.append(batch_mean_dists_from_models)

    all_models_mean_probs = np.concatenate(all_samples_probs_from_models, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)

    # Soft voting
    soft_vote_probs = np.mean(all_models_mean_probs, axis=1)
    soft_vote_preds = np.argmax(soft_vote_probs, axis=1)

    # Uncertainty metrics (see paper Section 3.4)
    # uncert_mc: within-model MC Dropout variance (variance across the M stochastic
    # passes, averaged over fold models). uncert_ensvar: variance across the fold
    # models' mean predictions, kept for reference.
    all_models_mcvar = np.concatenate(all_samples_mcvar_from_models, axis=0)  # [N, n_models]
    soft_vote_uncertainty_mc = np.mean(all_models_mcvar, axis=1)
    soft_vote_uncertainty_ensvar = np.mean(np.var(all_models_mean_probs, axis=1), axis=1)
    epsilon = 1e-10
    soft_vote_uncertainty_entropy = -np.sum(soft_vote_probs * np.log(soft_vote_probs + epsilon), axis=1)
    soft_vote_uncertainty_1minusmax = 1.0 - np.max(soft_vote_probs, axis=1)

    soft_acc = accuracy_score(all_labels, soft_vote_preds) * 100
    try:
        soft_auc = roc_auc_score(all_labels, soft_vote_probs, multi_class='ovr', average='macro')
    except ValueError:
        soft_auc = 0.0
    soft_f1 = f1_score(all_labels, soft_vote_preds, average='macro')

    soft_cm = confusion_matrix(all_labels, soft_vote_preds)
    soft_sensitivity, soft_specificity = calculate_sensitivity_specificity(soft_cm, classes)

    all_models_preds = np.argmax(all_models_mean_probs, axis=2)
    hard_vote_preds = stats.mode(all_models_preds, axis=1)[0].flatten()

    hard_acc = accuracy_score(all_labels, hard_vote_preds) * 100
    hard_f1 = f1_score(all_labels, hard_vote_preds, average='macro')
    hard_cm = confusion_matrix(all_labels, hard_vote_preds)
    hard_sensitivity, hard_specificity = calculate_sensitivity_specificity(hard_cm, classes)
    class_counts = [np.sum(all_labels == i) for i in range(len(classes))]

    print(f"\n{'='*80}\n5-fold ensemble test results (Temperature Scaling + MC Dropout)\n{'='*80}")
    print(f"\nSoft Voting (with Temperature Calibration):")
    print(f"  - Accuracy: {soft_acc:.2f}%")
    print(f"  - AUC: {soft_auc:.4f}")
    print(f"  - Macro F1: {soft_f1:.4f}")

    print("\nSoft Voting - Confusion Matrix:")
    print(soft_cm)
    print_sensitivity_specificity(soft_sensitivity, soft_specificity, classes, class_counts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_filename = f'ensemble_results_{timestamp}.npz'

    np.savez(npz_filename,
             y_true=all_labels,
             y_pred=soft_vote_preds,
             uncert_mc=soft_vote_uncertainty_mc,
             uncert_entropy=soft_vote_uncertainty_entropy,
             uncert_1minusmax=soft_vote_uncertainty_1minusmax,
             uncert_ensvar=soft_vote_uncertainty_ensvar)

    print(f"\nEnsemble results of this run (labels, predictions, uncertainties) saved to: {npz_filename}")
    print(f"  (can be aggregated over multiple runs to plot a smoother Accuracy-Rejection curve)")
    print(f"{'='*80}")

def run_training_fixed(gpu, world_size, fold_pairs, test_pairs, port, data_dir, classes,
                       focal_gamma=2.0, prototype_compact_weight=0.2, prototype_separation_weight=0.02,
                       epochs=80, batch_size=16, base_lr=1e-4, output_dir='results', mc_iterations=50):

    preload_list = list(test_pairs)
    for dest in [f'FOLD{i}' for i in range(1, 6)]:
        preload_list.extend(fold_pairs.get(dest, []))
    preload_data(data_dir, preload_list)

    if world_size > 1:
        setup(gpu, world_size, port)

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if gpu == 0 else None

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        g = torch.Generator()
        g.manual_seed(42)

        base_batch_size = batch_size
        num_workers = 2
        model_name = "prosyn_net"

        lr = base_lr * world_size if world_size > 1 else base_lr

        if gpu == 0:
            mode_str = "DDP" if world_size > 1 else "Single GPU"
            print(f"Mode: {mode_str}, learning rate: {lr:.1e}")
            print(f"Loss: Prototype Net Loss (gamma={focal_gamma}, compact={prototype_compact_weight}, separation={prototype_separation_weight})")

        if gpu == 0:
            print(f"Class names: {classes}")

        test_dataset = UltrasoundDataset(test_pairs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=base_batch_size * world_size if world_size > 1 else base_batch_size,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )

        fold_names = [f'FOLD{i}' for i in range(1, 6)]
        fold_results = []
        fold_temperatures = []

        for fold_idx, fold_name in enumerate(fold_names, start=1):
            if gpu == 0:
                print(f"\n{'='*80}\n{fold_name} - ProSyn-Net\n{'='*80}")

            val_pairs = fold_pairs.get(fold_name, [])
            train_pairs = []
            for other in fold_names:
                if other != fold_name:
                    train_pairs.extend(fold_pairs.get(other, []))

            # Class weights from THIS fold's training splits only, so the
            # validation/test label distributions never enter training
            class_counts = [max(sum(1 for p in train_pairs if p['label'] == i), 1) for i in range(len(classes))]
            total_samples = sum(class_counts)
            class_weights = torch.tensor([total_samples / (len(classes) * count) for count in class_counts])
            if gpu == 0:
                print(f"Class counts (train folds only): {class_counts}")

            train_dataset = UltrasoundDataset(train_pairs, augment=True)
            val_dataset = UltrasoundDataset(val_pairs)

            if world_size > 1:
                train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=gpu, seed=42)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=base_batch_size, num_workers=num_workers,
                    drop_last=True, sampler=train_sampler, persistent_workers=True,
                    pin_memory=True, worker_init_fn=seed_worker, generator=g
                )
            else:
                train_sampler = None
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=base_batch_size, num_workers=num_workers,
                    shuffle=True, persistent_workers=True, pin_memory=True,
                    worker_init_fn=seed_worker, generator=g
                )

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=base_batch_size, num_workers=num_workers,
                persistent_workers=True, pin_memory=True, worker_init_fn=seed_worker, generator=g
            )

            model = get_model('prototype_enhanced', num_classes=len(classes))

            if world_size > 1:
                model = model.to(gpu)
                ddp_model = DDP(model, device_ids=[gpu])
            else:
                ddp_model = model.to(gpu)

            trainer = ModelTrainer(gpu, world_size)
            trainer.current_fold = fold_idx
            _, best_model, temperature = trainer.train_model(
                ddp_model, train_loader, val_loader, train_sampler,
                epochs=epochs, lr=lr, model_name=model_name,
                focal_gamma=focal_gamma,
                prototype_compact_weight=prototype_compact_weight,
                prototype_separation_weight=prototype_separation_weight,
                class_weights=class_weights
            )

            if world_size > 1:
                dist.barrier()

            if gpu == 0:
                fold_temperatures.append(temperature)
                test_metrics = trainer.test_model(best_model, test_loader, classes,
                                                  temperature=temperature, mc_iterations=mc_iterations)
                fold_results.append(test_metrics)

        if gpu == 0:
            accs = [r['accuracy'] for r in fold_results]
            avg_acc = np.mean(accs)

            print(f"\n{'='*79}\n5-fold cross-validation summary\n{'='*79}")
            print(f"Mean accuracy: {avg_acc:.2f}% ± {np.std(accs):.2f}%")

            results_dir = output_dir
            os.makedirs(results_dir, exist_ok=True)

            folder_name = f"{timestamp}_prosyn_net_{avg_acc:.2f}"
            result_folder = os.path.join(results_dir, folder_name)
            os.makedirs(result_folder, exist_ok=True)

            model_paths_and_temps = []
            for i in range(1, 6):
                src_model = f'best_{model_name}_fold{i}_with_temp.pth'
                dst_model = os.path.join(result_folder, f'fold{i}_with_temp.pth')
                if os.path.exists(src_model):
                    shutil.copy2(src_model, dst_model)
                    model_paths_and_temps.append((src_model, fold_temperatures[i-1]))

            print(f"\n5-fold models saved to: {result_folder}")

            run_ensemble_test(model_paths_and_temps, test_loader, torch.device(f'cuda:{gpu}'),
                              classes, mc_iterations=mc_iterations)

    finally:
        if world_size > 1:
            cleanup()

def main():
    parser = argparse.ArgumentParser(description="ProSyn-Net training script")
    parser.add_argument('--data_dir', type=str, default='processed_data',
                        help="Root directory containing FOLD1..FOLD5 and TEST splits")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Directory where the per-run result folder is created")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mc_iterations', type=int, default=50,
                        help="Number of MC Dropout stochastic forward passes")
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--prototype_compact_weight', type=float, default=0.2)
    parser.add_argument('--prototype_separation_weight', type=float, default=0.02)
    args = parser.parse_args()

    processed_dir = args.data_dir

    if not os.path.isdir(processed_dir):
        print(f"Error: data directory not found: {processed_dir}")
        return

    fold_pairs, test_pairs, classes = load_processed_data(processed_dir)
    total_pairs = sum(len(v) for v in fold_pairs.values()) + len(test_pairs)
    if total_pairs == 0:
        print("Error: no samples were loaded.")
        return
    if len(classes) == 0:
        print("Error: no class directories found (expected subdirectories under TEST/ or FOLD1/).")
        return

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPU detected.")
        return

    if world_size > 1:
        port = find_free_port()
        mp.spawn(run_training_fixed,
                 args=(world_size, fold_pairs, test_pairs, port, processed_dir, classes,
                       args.focal_gamma, args.prototype_compact_weight, args.prototype_separation_weight,
                       args.epochs, args.batch_size, args.lr, args.output_dir, args.mc_iterations),
                 nprocs=world_size, join=True)
    else:
        run_training_fixed(0, 1, fold_pairs, test_pairs, 0, processed_dir, classes,
                           args.focal_gamma, args.prototype_compact_weight, args.prototype_separation_weight,
                           args.epochs, args.batch_size, args.lr, args.output_dir, args.mc_iterations)

if __name__ == "__main__":
    main()
