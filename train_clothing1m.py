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
from typing import Tuple, Optional, Callable, List, Sequence, Iterator, Dict

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, Sampler

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

# ---------------------------
# Your transforms (unchanged)
# ---------------------------
mean = (0.485, 0.456, 0.406)  # (0.6959, 0.6537, 0.6371)
std = (0.229, 0.224, 0.225)   # (0.3113, 0.3192, 0.3214)
normalize_tf = transforms.Normalize(mean=mean, std=std)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize_tf,
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# --------------------------------------
# Dataset updated to accept a transform
# --------------------------------------
class ClothingNPZDataset(Dataset):
    """
    Memory-friendly loader for .npz with arr_0 (images, HWC) and arr_1 (labels).
    Optionally resizes tensors to a target square size using bilinear interpolation.

    If `transform` is provided, it is applied to a PIL image produced from the raw
    HWC numpy array. In that case, `normalize` and `resize_to` inside the dataset
    are bypassed for the image (handled by the transform pipeline).
    """
    def __init__(self, npz_path: str, normalize: bool = True, mmap_try: bool = True,
                 resize_to: int = 0, device: Optional[torch.device] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.npz_path = npz_path
        self.normalize = normalize
        self.device = device  # tensors moved in the train loop
        self.resize_to = int(resize_to) if resize_to else 0
        self.transform = transform
        self.target_transform = target_transform

        mmap_mode = 'r' if mmap_try else None
        self.archive = np.load(npz_path, allow_pickle=False, mmap_mode=mmap_mode)
        if 'arr_0' not in self.archive or 'arr_1' not in self.archive:
            raise ValueError(f"{npz_path} must contain 'arr_0' (images) and 'arr_1' (labels).")

        self.images = self.archive['arr_0']  # (N, H, W, 3), dtype uint8 or float
        self.labels = self.archive['arr_1']  # (N,)

        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"Expected images of shape (N, H, W, 3), got {self.images.shape}.")

        # Only used when self.transform is None
        self.registered_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.registered_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        img = self.images[idx]  # HWC
        label = int(self.labels[idx])

        if self.transform is not None:
            # Ensure uint8 HWC for PIL conversion
            #if np.issubdtype(img.dtype, np.floating):
            #    if img.max() <= 1.0 + 1e-6:
            #        img_np = (img * 255.0).round().astype(np.uint8)
            #    else:
            #        img_np = np.clip(img, 0, 255).round().astype(np.uint8)
            #else:
                
            img_np = img.astype(np.uint8, copy=False)

            pil_img = Image.fromarray(img_np)  # assumes RGB
            img_t = self.transform(pil_img)    # e.g., train_transform / transform_test
        else:
            # Original tensor path (kept for compatibility)
            if img.dtype != np.float32 and img.dtype != np.float64:
                img = img.astype(np.float32) / 255.0
            else:
                if img.max() > 1.0:
                    img = img / 255.0

            img_t = torch.from_numpy(img).permute(2, 0, 1)  # CHW, float in [0,1]

            if self.normalize:
                img_t = (img_t - self.registered_mean) / self.registered_std

            if self.resize_to and (img_t.shape[1] != self.resize_to or img_t.shape[2] != self.resize_to):
                img_t = F.interpolate(img_t.unsqueeze(0), size=(self.resize_to, self.resize_to),
                                      mode='bilinear', align_corners=False).squeeze(0)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img_t, label


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """
    Yields `n_batches` batches with class-balanced composition.
    - If batch_size is divisible by #classes: exact equality per batch.
    - Otherwise: per-batch counts differ by at most 1. The 'extra' samples are
      assigned to different classes each batch using a rotating schedule.

    Sampling is with replacement (robust to tiny classes).

    Args:
        labels: sequence of dataset labels (len == len(dataset))
        batch_size: size of each mini-batch
        n_batches: number of mini-batches per epoch
        seed: optional seed for reproducibility
        shuffle_within_batch: shuffle indices inside each batch
        remainder_strategy: 'rotate' (default) or 'random'
            - 'rotate' fairly rotates which classes get +1 across batches
            - 'random' picks r classes uniformly at random per batch
    """
    def __init__(
        self,
        labels: Sequence,
        batch_size: int,
        n_batches: int,
        seed: Optional[int] = None,
        shuffle_within_batch: bool = True,
        remainder_strategy: str = "rotate",
    ):
        self.labels_np = np.asarray(labels)
        self.classes = np.sort(np.unique(self.labels_np))
        self.n_classes = len(self.classes)
        if self.n_classes < 1:
            raise ValueError("No classes found in labels.")

        self.batch_size = int(batch_size)
        self.n_batches = int(n_batches)
        self.shuffle_within_batch = shuffle_within_batch
        self.remainder_strategy = remainder_strategy
        self.seed = seed
        self._epoch = 0

        # Precompute index lists for each class
        self.class_to_indices: Dict = {
            c: np.where(self.labels_np == c)[0] for c in self.classes
        }
        for c, idxs in self.class_to_indices.items():
            if idxs.size == 0:
                raise ValueError(f"Class {c} has no samples.")

        # Base-per-class and per-batch remainder
        self.base_per_class = self.batch_size // self.n_classes
        self.remainder = self.batch_size - self.base_per_class * self.n_classes

        # For 'rotate' strategy we maintain a circular permutation of class ids
        self._perm = np.arange(self.n_classes)

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to alter RNG deterministically."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(None if self.seed is None else self.seed + self._epoch)

        # Prepare rotating order for remainder assignment
        perm = self._perm.copy()
        rng.shuffle(perm)
        rotate_ptr = 0  # start position in the circular permutation

        for _ in range(self.n_batches):
            batch: List[int] = []

            # 1) Base equal count for every class
            if self.base_per_class > 0:
                for c in self.classes:
                    pool = self.class_to_indices[c]
                    chosen = rng.choice(pool, size=self.base_per_class, replace=True)
                    batch.extend(chosen.tolist())

            # 2) Distribute the remainder (+1) to `remainder` classes
            if self.remainder > 0:
                if self.remainder_strategy == "rotate":
                    # take a window of size `remainder` from the circular perm
                    window_idx = (np.arange(rotate_ptr, rotate_ptr + self.remainder)) % self.n_classes
                    remainder_class_indices = perm[window_idx]
                    rotate_ptr = (rotate_ptr + self.remainder) % self.n_classes
                elif self.remainder_strategy == "random":
                    remainder_class_indices = rng.choice(self.n_classes, size=self.remainder, replace=False)
                else:
                    raise ValueError("remainder_strategy must be 'rotate' or 'random'.")

                for ci in remainder_class_indices:
                    c = self.classes[int(ci)]
                    pool = self.class_to_indices[c]
                    extra = rng.choice(pool, size=1, replace=True)
                    batch.append(int(extra[0]))

            if self.shuffle_within_batch:
                rng.shuffle(batch)

            yield batch

def sample_traning_set(train_imgs, labels, num_class, num_samples):
    random.shuffle(train_imgs)
    class_num = torch.zeros(num_class)
    sampled_train_imgs = []
    for impath in train_imgs:
        label = labels[impath]
        if class_num[label] < (num_samples / num_class):
            sampled_train_imgs.append(impath)
            class_num[label] += 1
        if len(sampled_train_imgs) >= num_samples:
            break
    return sampled_train_imgs

class clothing_dataset(Dataset):
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14,
                 add_clean=False, log=None, clean_all=False):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.noisy_labels = {}
        self.clean_labels = {}
        self.val_labels = {}
        self.clean_all = clean_all
        #paths = eval_train()

        with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.noisy_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/' % self.root + entry[0][7:]
                self.clean_labels[img_path] = int(entry[1])
        # Clean size: 72409. Noisy size: 1037497. Clean/noisy intersection: 37497 (24637/5395/7465)

        if self.mode == "labeled":
            self.train_imgs = paths[:num_samples]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

        elif mode == 'all':
            train_imgs = []
            with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    train_imgs.append(img_path)
            self.train_imgs = sample_traning_set(train_imgs, self.noisy_labels, num_class, num_samples)
            random.shuffle(self.train_imgs)
            if add_clean:
                inter_imgs = []
                for impath in self.clean_labels:
                    if impath in self.noisy_labels:
                        inter_imgs.append(impath)
                self.train_imgs += inter_imgs  # add images which have a clean label too to be able to calculate metrics
        
        elif mode == 'test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.test_imgs.append(img_path)

        elif mode == 'val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/' % self.root + l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.noisy_labels[img_path]
            #prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            #img2 = self.transform(image)
            return img, target #, 0, prob
        #elif self.mode == 'unlabeled':
        #    img_path = self.train_imgs[index]
        #    image = Image.open(img_path).convert('RGB')
        #    img = self.transform(image)
        #    img2 = self.transform(image)
        #    return img
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.noisy_labels[img_path] if not self.clean_all else self.clean_labels[img_path]
            clean_target = self.clean_labels.get(img_path, -1)
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            #img2 = self.transform(image)
            return img, target, img_path, clean_target
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.clean_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.clean_labels[img_path] #self.noisy_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        if self.mode == 'val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)

def eval_train(eval_loader, num_batches, batch_size):
    #model.eval()
    num_samples = num_batches * batch_size + 37497  # add for intersection
    #losses = torch.zeros(num_samples)
    paths = []
    #n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path, clean_target) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            #outputs = model(inputs)
            #loss = criterion(outputs, targets)
            for b in range(inputs.size(0)):
                #losses[n] = loss[b]
                paths.append(path[b])
                n += 1
            
    return paths

class clothing_dataloader():
    def __init__(self, root, batch_size, num_batches, num_workers, log=None, stronger_aug=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root
        self.log = log

        mean = (0.485, 0.456, 0.406)  # (0.6959, 0.6537, 0.6371),
        std = (0.229, 0.224, 0.225)  # (0.3113, 0.3192, 0.3214)
        normalize = transforms.Normalize(mean=mean, std=std)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_warmup = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomAffine(degrees=15.,
                                    translate=(0.1, 0.1),
                                    scale=(2. / 3, 3. / 2),
                                    shear=(-0.1, 0.1, -0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_train = train_transform
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.transform_warmup = self.transform_warmup if stronger_aug else self.transform_train
        self.warmup_samples = self.num_batches * self.batch_size * 4 if stronger_aug else self.num_batches * self.batch_size * 2

    def run(self, mode, pred=[], prob=[], paths=[]):
        if mode == 'train':
            labeled_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='labeled', paths=paths,
                                               num_samples=self.num_batches * self.batch_size, log=self.log)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            
            return labeled_loader
        elif mode == 'eval_train':
            eval_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='all',
                                            num_samples=self.num_batches * self.batch_size, add_clean=True,
                                            log=self.log)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
        elif mode == 'test':
            test_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='test', log=self.log)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
        elif mode == 'val':
            val_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='val', log=self.log)
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return val_loader



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
        per_sample_ce = - torch.sum(torch.log(softmax_outputs) * encoded_targets, dim=1)

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
    parser.add_argument("--num_batches", type=int, default=1000)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model name (default: resnet50)")
    parser.add_argument("--pretrained", type=lambda x: str(x).lower() in ["1","true","yes","y","t"], default=True)
    parser.add_argument("--loss", type=str, default="CE", choices=["CE", "FL"])
    parser.add_argument("--use_lilaw", type=lambda x: str(x).lower() in ["1","true","yes","y","t"], default=True)
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Epochs before applying LiLAW updates")
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
    loader = clothing_dataloader(root='/home/moturuab/projects/aip-agoldenb/moturuab/clothing1M', 
        batch_size=args.batch_size, num_workers=5, num_batches=args.num_batches)

    #labels_arr = train_full.labels
    #num_classes = int(np.max(labels_arr)) + 1
    #print(f"[INFO] Detected {num_classes} classes.")
    eval_loader = loader.run('eval_train')
    paths = eval_train(eval_loader, args.num_batches, args.batch_size)
    train_loader = loader.run('train', paths)
    val_loader = loader.run('val')
    test_loader = loader.run('test')

    #def meta_iter_fn():
    #    while True:
    #        for batch in meta_loader:
    #            yield batch
    #meta_iter = meta_iter_fn()

    # ---------------------------
    # Model, optimizer, scaler
    # ---------------------------
    model = build_model(args.model_name, num_classes=num_classes, pretrained=args.pretrained).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.use_amp))

    # LiLAW parameters as learnable scalars (updated via meta step)
    alpha = nn.Parameter(torch.tensor(args.alpha_init, dtype=torch.float32, device=device), requires_grad=True)
    beta  = nn.Parameter(torch.tensor(args.beta_init,  dtype=torch.float32, device=device), requires_grad=True)
    delta = nn.Parameter(torch.tensor(args.delta_init, dtype=torch.float32, device=device), requires_grad=True)

    if args.use_lilaw:
        optimizer = optim.AdamW(list(model.parameters()) + list([alpha, beta, delta]), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
                "num_batches": args.num_batches,
                "lr": args.lr,
                "meta_fraction": args.meta_fraction,
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
            model.train()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True, dtype=torch.long)

            # TRAIN STEP (update model only)
            for p in model.parameters():
                p.grad = None

            alpha.grad = None
            beta.grad = None
            delta.grad = None

            model.requires_grad = True
            for p in model.parameters():
                p.requires_grad = True
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

            for p in model.parameters():
                p.grad = None

            alpha.grad = None
            beta.grad = None
            delta.grad = None

            model.eval()

            # META-VALIDATION STEP (update LiLAW scalars per train batch)
            if args.use_lilaw and epoch > args.warmup_epochs:
                #meta_images, meta_labels = next(meta_iter)
                for meta_images, meta_labels in val_loader:
                    meta_images = meta_images.to(device, non_blocking=True)
                    meta_labels = meta_labels.to(device, non_blocking=True, dtype=torch.long)

                    alpha.requires_grad = True
                    beta.requires_grad = True
                    delta.requires_grad = True
                    model.requires_grad = False
                    for p in model.parameters():
                        p.requires_grad = False

                    #with torch.no_grad():
                    meta_logits = model(meta_images)

                    _, _, _, _, _, _, meta_loss = criterion(meta_logits, meta_labels, epoch=epoch)

                    #if alpha.grad is not None: alpha.grad.zero_()
                    #if beta.grad is not None:  beta.grad.zero_()
                    #if delta.grad is not None: delta.grad.zero_()

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
                        p.grad = None

                    break

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
