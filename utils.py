import torch
try:
    from torchvision import datasets, transforms
except Exception:
    datasets = None
    transforms = None
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


# ─────────────────────────────────────────────
#  MNIST
# ─────────────────────────────────────────────

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    testset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return trainset, testset


def create_non_iid_splits(dataset, num_clients=3):
    """Non-IID split for MNIST by digit class groups."""
    from torch.utils.data import Subset

    # 3-client split: digits 0-3 | 4-6 | 7-9
    buckets = [[], [], []]
    for i, (_, label) in enumerate(dataset):
        if label in [0, 1, 2, 3]:
            buckets[0].append(i)
        elif label in [4, 5, 6]:
            buckets[1].append(i)
        else:
            buckets[2].append(i)

    subsets = [Subset(dataset, idx) for idx in buckets]

    print("\n[NON-IID VERIFICATION] MNIST")
    for c, b in enumerate(buckets):
        print(f"  Client {c+1}: {len(b):,} samples")

    return subsets


# ─────────────────────────────────────────────
#  HEART  (FIX: StandardScaler applied here)
# ─────────────────────────────────────────────

# Module-level scaler so test set uses the same fit
_heart_scaler = StandardScaler()


def load_heart_data(csv_path="./data/heart.csv", test_size=0.2, random_state=42):
    """Load and SCALE heart data.

    ROOT CAUSE FIX:
    - Heart features are unscaled (mean~46, std~81)
    - Large inputs → sigmoid saturates near 0 → BCELoss explodes to ~21
    - StandardScaler fixes this: outputs become healthy (0.1–0.9)
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    if df.shape[1] != 14:
        raise ValueError(f"Expected 14 columns, got {df.shape[1]}")

    data = df.values
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)

    # Binarise labels: some CSVs have 0–4, treat >0 as disease
    y = (y > 0).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ── KEY FIX: fit scaler on train, apply to both ──
    X_train = _heart_scaler.fit_transform(X_train).astype(np.float32)
    X_test  = _heart_scaler.transform(X_test).astype(np.float32)

    print(f"[INFO] Heart data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"[INFO] Heart features after scaling — mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test,  dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test,  dtype=torch.float32),
    )


def create_non_iid_splits_heart(X_tensor, y_tensor, num_clients=3):
    """Non-IID split for Heart: each client gets skewed class ratio."""
    from torch.utils.data import TensorDataset

    y_np = y_tensor.numpy()
    c0 = np.where(y_np == 0)[0]
    c1 = np.where(y_np == 1)[0]

    rng = np.random.RandomState(42)
    rng.shuffle(c0)
    rng.shuffle(c1)

    c0_parts = np.array_split(c0, num_clients)
    c1_parts = np.array_split(c1, num_clients)

    print("\n[NON-IID VERIFICATION] Heart Disease")
    subsets = []
    for i in range(num_clients):
        p0, p1 = c0_parts[i], c1_parts[i]
        # Skew: client 0 → mostly class 0, client 2 → mostly class 1
        if i == 0:
            idxs = np.concatenate([p0, p1[:max(1, len(p1)//4)]])
        elif i == num_clients - 1:
            idxs = np.concatenate([p0[:max(1, len(p0)//4)], p1])
        else:
            idxs = np.concatenate([p0, p1])

        rng.shuffle(idxs)
        xi = X_tensor[idxs]
        yi = y_tensor[idxs].to(dtype=torch.float32)

        c0_count = int((yi == 0).sum())
        c1_count = int((yi == 1).sum())
        print(f"  Client {i+1}: {len(idxs)} samples | class-0: {c0_count}, class-1: {c1_count}")

        subsets.append(TensorDataset(xi, yi))

    return subsets


# ─────────────────────────────────────────────
#  DIABETES  (FIX: normalise y target)
# ─────────────────────────────────────────────

_diabetes_y_mean = None
_diabetes_y_std  = None


def load_diabetes_data():
    """Load and normalise diabetes data.

    ROOT CAUSE FIX:
    - y target ranges 25–346 → MSELoss huge → gradients explode
    - Normalise y to mean=0, std=1 for stable training
    - Denormalise at evaluation for real RMSE
    """
    global _diabetes_y_mean, _diabetes_y_std

    data = load_diabetes()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalise y (X is already normalised by sklearn)
    _diabetes_y_mean = float(y_train.mean())
    _diabetes_y_std  = float(y_train.std()) + 1e-8

    y_train_norm = (y_train - _diabetes_y_mean) / _diabetes_y_std
    y_test_norm  = (y_test  - _diabetes_y_mean) / _diabetes_y_std

    print(f"[INFO] Diabetes data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"[INFO] y normalised — mean: {_diabetes_y_mean:.2f}, std: {_diabetes_y_std:.2f}")

    return (
        torch.tensor(X_train,      dtype=torch.float32),
        torch.tensor(X_test,       dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32),
        torch.tensor(y_test_norm,  dtype=torch.float32),
        _diabetes_y_mean,
        _diabetes_y_std,
    )


def create_non_iid_splits_diabetes(X_tensor, y_tensor, num_clients=3):
    """Non-IID split for Diabetes by target quantile ranges."""
    from torch.utils.data import TensorDataset

    y_np = y_tensor.numpy()
    quantiles = np.quantile(y_np, np.linspace(0, 1, num_clients + 1))

    print("\n[NON-IID VERIFICATION] Diabetes")
    subsets = []
    for i in range(num_clients):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask = (y_np >= lo) & (y_np <= hi) if i == num_clients - 1 else (y_np >= lo) & (y_np < hi)
        xi, yi = X_tensor[mask], y_tensor[mask]
        print(f"  Client {i+1}: {int(mask.sum())} samples | y range [{lo:.3f}, {hi:.3f}]")
        subsets.append(TensorDataset(xi, yi))

    return subsets


def get_diabetes_y_stats():
    """Return (mean, std) used to normalise y — needed for denormalised RMSE."""
    return _diabetes_y_mean, _diabetes_y_std
