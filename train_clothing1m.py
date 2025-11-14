"""
train_clothing1m_lilaw_resnet50.py

Single-file PyTorch pipeline for Clothing-1M (64x64 RGB) + Clothing-10k test.
NOW DEFAULTS TO IMAGENET-PRETRAINED RESNET-50 (fine-tuned).

Datasets:
  - training+meta-val: clothing1m.npz
      d['arr_0']: RGB images, shape (1000000, 64, 64, 3)
      d['arr_1']: int labels, shape (1000000,)
  - test: clothing10k_test.npz
      d['arr_0']: RGB images, shape (10526, 64, 64, 3)
      d['arr_1']: int labels, shape (10526,)

Features:
  - Split: train 85%, meta-validation 15% (shuffled once with seed)
  - LiLAW reweighting (on/off via --use_lilaw) with meta update per train step
  - CUDA + AMP, tqdm, wandb, checkpoints
  - Optional resize (disabled by default to keep memory in check)

LiLAW weighting logic adapted from your provided reference file.  # (Cited in the message)
"""

import os
import math
import time
import random
import argparse
from typing import Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# Optional deps
try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False


# ---------------------------
# Utilities / Repro
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN fast (not strictly deterministic)
    torch.backends.cudnn.benchmark = True


# ---------------------------
# Dataset for NPZ (+ optional resize)
# ---------------------------

class ClothingNPZDataset(Dataset):
    """
    Memory-friendly loader for .npz with arr_0 (images, HWC) and arr_1 (labels).
    Optionally resizes tensors to a target square size using bilinear interpolation.
    """
    def __init__(self, npz_path: str, normalize: bool = True, mmap_try: bool = True,
                 resize_to: int = 0, device: Optional[torch.device] = None):
        self.npz_path = npz_path
        self.normalize = normalize
        self.device = device  # unused here; tensors are moved in the train loop
        self.resize_to = int(resize_to) if resize_to else 0

        mmap_mode = 'r' if mmap_try else None
        self.archive = np.load(npz_path, allow_pickle=False, mmap_mode=mmap_mode)
        if 'arr_0' not in self.archive or 'arr_1' not in self.archive:
            raise ValueError(f"{npz_path} must contain 'arr_0' (images) and 'arr_1' (labels).")

        self.images = self.archive['arr_0']  # (N, 64, 64, 3), dtype uint8 or float
        self.labels = self.archive['arr_1']  # (N,)

        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"Expected images of shape (N, H, W, 3), got {self.images.shape}.")

        self.registered_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.registered_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = int(self.labels[idx])

        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        else:
            if img.max() > 1.0:
                img = img / 255.0

        img_t = torch.from_numpy(img).permute(2, 0, 1)  # CHW

        if self.normalize:
            img_t = (img_t - self.registered_mean) / self.registered_std

        if self.resize_to and (img_t.shape[1] != self.resize_to or img_t.shape[2] != self.resize_to):
            # Resize CHW tensor
            img_t = F.interpolate(img_t.unsqueeze(0), size=(self.resize_to, self.resize_to),
                                  mode='bilinear', align_corners=False).squeeze(0)

        return img_t, label


# ---------------------------
# (Fallback) Simple CNN (kept only as a safety net)
# ---------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# Build model: DEFAULT = ResNet-50 (ImageNet pretrained)
# ---------------------------

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    name = model_name.lower().replace("-", "")
    if name == "resnet50":
        # Prefer timm if available; otherwise fall back to torchvision
        if HAS_TIMM:
            model = timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
            return model
        else:
            try:
                from torchvision.models import resnet50, ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                model = resnet50(weights=weights)
                # Replace classification head
                in_feats = model.fc.in_features
                model.fc = nn.Linear(in_feats, num_classes)
                return model
            except Exception as e:
                print(f"[WARN] torchvision not available ({e}); falling back to a simple CNN (not pretrained).")
                return SimpleCNN(num_classes)

    # If another model was explicitly requested:
    if HAS_TIMM:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    print("[WARN] Requested model not found and timm not available; using SimpleCNN.")
    return SimpleCNN(num_classes)


# ---------------------------
# LiLAW Losses (adapted to single-file use)
# ---------------------------

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, reweight=True, alpha=None, beta=None, delta=None,
                 num_classes=2, warmup=0, device=None, model_name=None):
        super().__init__()
        self.reweight = reweight
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.num_classes = int(num_classes)
        self.warmup = int(warmup)
        self.device = device
        self.model_name = model_name
        self.sigmoid = nn.Sigmoid()

    def encode(self, targets):
        encoded_targets = torch.zeros(targets.size(0), self.num_classes, device=targets.device, dtype=torch.float32)
        encoded_targets.scatter_(1, targets.view(-1, 1).long(), 1.0)
        return encoded_targets

    def _weights(self, correct_outputs, max_outputs):
        alpha_weights = self.sigmoid(self.alpha * correct_outputs - max_outputs)
        beta_weights  = self.sigmoid(-(self.beta * correct_outputs - max_outputs))
        delta_weights = torch.exp(- (-(self.delta * correct_outputs - max_outputs)) ** 2 / 2)
        weights = alpha_weights + beta_weights + delta_weights
        return alpha_weights, beta_weights, delta_weights, weights

    def forward(self, outputs, targets, epoch: int = -1):
        softmax_outputs = F.softmax(outputs, dim=1)
        encoded_targets = self.encode(targets)
        per_sample_ce = - torch.sum(torch.log(softmax_outputs + 1e-12) * encoded_targets, dim=1)

        correct_outputs = softmax_outputs.gather(1, torch.argmax(encoded_targets, dim=1).unsqueeze(1)).squeeze(1)
        max_outputs     = softmax_outputs.gather(1, torch.argmax(softmax_outputs, dim=1).unsqueeze(1)).squeeze(1)

        if self.reweight and epoch > self.warmup:
            alpha_w, beta_w, delta_w, weights = self._weights(correct_outputs, max_outputs)
            weighted_loss = weights * per_sample_ce
            return correct_outputs.detach(), max_outputs.detach(), alpha_w.detach(), beta_w.detach(), delta_w.detach(), weights.detach(), weighted_loss.mean()
        else:
            return correct_outputs.detach(), max_outputs.detach(), None, None, None, None, per_sample_ce.mean()


class WeightedFocalLoss(nn.Module):
    def __init__(self, reweight=True, alpha=None, beta=None, delta=None, gamma=2.0,
                 num_classes=2, warmup=0, device=None, model_name=None):
        super().__init__()
        self.reweight = reweight
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.num_classes = int(num_classes)
        self.warmup = int(warmup)
        self.device = device
        self.model_name = model_name
        self.ce_helper = WeightedCrossEntropyLoss(
            reweight=False, alpha=alpha, beta=beta, delta=delta,
            num_classes=num_classes, warmup=0, device=device, model_name=model_name
        )

    def forward(self, outputs, targets, epoch: int = -1):
        softmax_outputs = F.softmax(outputs, dim=1)
        encoded_targets = self.ce_helper.encode(targets)
        per_sample_ce = - torch.sum(torch.log(softmax_outputs + 1e-12) * encoded_targets, dim=1)
        focal = (1.0 - torch.exp(-per_sample_ce)).pow(self.gamma) * per_sample_ce

        correct_outputs = softmax_outputs.gather(1, torch.argmax(encoded_targets, dim=1).unsqueeze(1)).squeeze(1)
        max_outputs     = softmax_outputs.gather(1, torch.argmax(softmax_outputs, dim=1).unsqueeze(1)).squeeze(1)

        if self.reweight and epoch > self.warmup:
            alpha_w, beta_w, delta_w, weights = self.ce_helper._weights(correct_outputs, max_outputs)
            weighted_focal = weights * focal
            return correct_outputs.detach(), max_outputs.detach(), alpha_w.detach(), beta_w.detach(), delta_w.detach(), weights.detach(), weighted_focal.mean()
        else:
            return correct_outputs.detach(), max_outputs.detach(), None, None, None, None, focal.mean()


# ---------------------------
# Training / Validation / Test
# ---------------------------

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / float(targets.size(0))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_correct = 0
    total_count = 0
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Clothing-1M LiLAW training with ResNet-50 (single-file).")
    parser.add_argument("--train_npz", type=str, default="clothing1m.npz", help="Path to clothing1m.npz")
    parser.add_argument("--test_npz", type=str, default="clothing10k_test.npz", help="Path to clothing10k_test.npz")
    parser.add_argument("--meta_fraction", type=float, default=0.15, help="Meta-val fraction of training set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model name (default: resnet50)")
    parser.add_argument("--pretrained", type=lambda x: str(x).lower() in ["1","true","yes","y","t"], default=True)
    parser.add_argument("--loss", type=str, default="CE", choices=["CE", "FL"])
    parser.add_argument("--use_lilaw", type=lambda x: str(x).lower() in ["1","true","yes","y","t"], default=True)
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Epochs before applying LiLAW updates")
    parser.add_argument("--alpha_init", type=float, default=10.0)
    parser.add_argument("--beta_init", type=float, default=2.0)
    parser.add_argument("--delta_init", type=float, default=6.0)
    parser.add_argument("--alpha_lr", type=float, default=5e-3)
    parser.add_argument("--beta_lr", type=float, default=5e-3)
    parser.add_argument("--delta_lr", type=float, default=5e-3)
    parser.add_argument("--alpha_wd", type=float, default=1e-4)
    parser.add_argument("--beta_wd", type=float, default=1e-4)
    parser.add_argument("--delta_wd", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma if loss=FL")
    parser.add_argument("--use_amp", type=lambda x: str(x).lower() in ["1","true","yes","y","t"], default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="example_difficulty")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--resize_to", type=int, default=0, help="0=keep 64x64; otherwise resize to this square side (e.g., 224)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------------------------
    # Load datasets and split
    # ---------------------------
    print("[INFO] Loading datasets...")
    train_full = ClothingNPZDataset(args.train_npz, normalize=True, mmap_try=True,
                                    resize_to=args.resize_to, device=device)
    test_ds = ClothingNPZDataset(args.test_npz, normalize=True, mmap_try=True,
                                 resize_to=args.resize_to, device=device)

    labels_arr = train_full.labels
    num_classes = int(np.max(labels_arr)) + 1
    print(f"[INFO] Detected {num_classes} classes.")

    N = len(train_full)
    meta_size = int(round(args.meta_fraction * N))
    train_size = N - meta_size
    indices = np.random.RandomState(args.seed).permutation(N)
    train_idx = indices[:train_size]
    meta_idx = indices[train_size:]
    print(f"[INFO] Train size: {train_size:,} | Meta-val size: {meta_size:,} | Test size: {len(test_ds):,}")

    train_ds = Subset(train_full, train_idx)
    meta_ds = Subset(train_full, meta_idx)

    val_bs = args.val_batch_size if args.val_batch_size is not None else args.batch_size

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=(args.num_workers > 0)
    )
    meta_loader = DataLoader(
        meta_ds, batch_size=val_bs, shuffle=True, pin_memory=True,
        num_workers=max(1, args.num_workers // 2), drop_last=True, persistent_workers=(args.num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_bs, shuffle=False, pin_memory=True,
        num_workers=max(1, args.num_workers // 2)
    )

    def meta_iter_fn():
        while True:
            for batch in meta_loader:
                yield batch
    meta_iter = meta_iter_fn()

    # ---------------------------
    # Model, optimizer, scaler
    # ---------------------------
    model = build_model(args.model_name, num_classes=num_classes, pretrained=args.pretrained).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.use_amp))

    # LiLAW parameters as learnable scalars (updated via meta step)
    alpha = nn.Parameter(torch.tensor(args.alpha_init, dtype=torch.float32, device=device), requires_grad=True)
    beta  = nn.Parameter(torch.tensor(args.beta_init, dtype=torch.float32, device=device), requires_grad=True)
    delta = nn.Parameter(torch.tensor(args.delta_init, dtype=torch.float32, device=device), requires_grad=True)

    # Criterion
    if args.loss.upper() == "CE":
        criterion = WeightedCrossEntropyLoss(
            reweight=args.use_lilaw, alpha=alpha, beta=beta, delta=delta,
            num_classes=num_classes, warmup=args.warmup_epochs, device=device, model_name=args.model_name
        )
    else:
        criterion = WeightedFocalLoss(
            reweight=args.use_lilaw, alpha=alpha, beta=beta, delta=delta, gamma=args.gamma,
            num_classes=num_classes, warmup=args.warmup_epochs, device=device, model_name=args.model_name
        )

    # ---------------------------
    # wandb (optional)
    # ---------------------------
    if HAS_WANDB:
        wandb_run = wandb.init(
            project=args.project,
            name=args.run_name,
            config={
                "model": args.model_name,
                "pretrained": args.pretrained,
                "loss": args.loss,
                "use_lilaw": args.use_lilaw,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_epochs": args.warmup_epochs,
                "alpha_init": args.alpha_init,
                "beta_init": args.beta_init,
                "delta_init": args.delta_init,
                "alpha_lr": args.alpha_lr,
                "beta_lr": args.beta_lr,
                "delta_lr": args.delta_lr,
                "alpha_wd": args.alpha_wd,
                "beta_wd": args.beta_wd,
                "delta_wd": args.delta_wd,
                "resize_to": args.resize_to,
            }
        )
        wandb.watch(model, log="all")
    else:
        wandb_run = None
        print("[INFO] wandb not available; proceeding without logging.")

    # ---------------------------
    # Train loop
    # ---------------------------
    global_step = 0
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_acc = 0.0
        ep_count = 0
        tbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        lr = args.lr
        if epoch >= 40:
            lr /= 10
        if epoch >= 80:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, (images, labels) in enumerate(tbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True, dtype=torch.long)

            # TRAIN STEP (update model only)
            alpha.requires_grad = False
            beta.requires_grad = False
            delta.requires_grad = False

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.use_amp)):
                logits = model(images)
                _, _, _, _, _, _, train_loss = criterion(logits, labels, epoch=epoch)

            # Backprop for model params
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            train_acc = accuracy_from_logits(logits.detach(), labels)
            ep_loss += train_loss.item() * labels.size(0)
            ep_acc += train_acc * labels.size(0)
            ep_count += labels.size(0)

            # META-VALIDATION STEP (update LiLAW scalars per train batch)
            if args.use_lilaw and epoch > args.warmup_epochs:
                alpha.requires_grad = True
                beta.requires_grad = True
                delta.requires_grad = True
                for p in model.parameters():
                    p.requires_grad = False

                meta_images, meta_labels = next(meta_iter)
                meta_images = meta_images.to(device, non_blocking=True)
                meta_labels = meta_labels.to(device, non_blocking=True, dtype=torch.long)

                with torch.no_grad():
                    meta_logits = model(meta_images)

                _, _, _, _, _, _, meta_loss = criterion(meta_logits, meta_labels, epoch=epoch)

                if alpha.grad is not None: alpha.grad.zero_()
                if beta.grad is not None:  beta.grad.zero_()
                if delta.grad is not None: delta.grad.zero_()

                meta_loss.backward()

                with torch.no_grad():
                    alpha -= args.alpha_lr * (alpha.grad + args.alpha_wd * alpha)
                    beta  -= args.beta_lr  * (beta.grad  + args.beta_wd  * beta)
                    delta -= args.delta_lr * (delta.grad + args.delta_wd * delta)

                    alpha.data.clamp_(min=1.0)
                    delta.data.clamp_(min=beta.detach().item())

                alpha.grad = None
                beta.grad = None
                delta.grad = None

                for p in model.parameters():
                    p.requires_grad = True

            # Logging per step
            if HAS_WANDB:
                log_dict = {
                    "train/loss": train_loss.item(),
                    "train/acc": train_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "global_step": global_step,
                }
                if args.use_lilaw:
                    log_dict.update({
                        "lilaw/alpha": float(alpha.detach().item()),
                        "lilaw/beta": float(beta.detach().item()),
                        "lilaw/delta": float(delta.detach().item()),
                    })
                wandb.log(log_dict)
            global_step += 1

            tbar.set_postfix({
                "loss": f"{(ep_loss/max(1,ep_count)):.4f}",
                "acc": f"{(ep_acc/max(1,ep_count)):.4f}",
                "α": f"{alpha.detach().item():.2f}" if args.use_lilaw else "-",
                "β": f"{beta.detach().item():.2f}" if args.use_lilaw else "-",
                "δ": f"{delta.detach().item():.2f}" if args.use_lilaw else "-",
            })

        # End epoch: test evaluation
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}] Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        if HAS_WANDB:
            wandb.log({
                "test/loss": test_loss,
                "test/acc": test_acc,
                "epoch": epoch
            })

        # Save checkpoint(s)
        if (epoch % args.save_every) == 0 and False:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "alpha": alpha.detach().item(),
                "beta": beta.detach().item(),
                "delta": delta.detach().item(),
                "args": vars(args),
            }, ckpt_path)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "alpha": alpha.detach().item(),
                "beta": beta.detach().item(),
                "delta": delta.detach().item(),
                "best_test_acc": best_test_acc,
                "args": vars(args),
            }, best_path)

    print(f"[DONE] Best test acc: {best_test_acc:.4f}")
    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
