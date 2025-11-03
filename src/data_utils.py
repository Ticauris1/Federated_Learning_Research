import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.auto import tqdm
import torch

def find_image_files(root_directory):
    """Scans a directory for image files and extracts their paths and class names."""
    image_paths, labels = [], []
    if not os.path.isdir(root_directory):
        print(f"❌ ERROR: Directory not found at {root_directory}")
        return image_paths, labels
    for dirpath, _, filenames in os.walk(root_directory):
        if os.path.basename(dirpath).startswith('.'):
            continue
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(dirpath, filename))
                labels.append(os.path.basename(dirpath))
    print(f"[{os.path.basename(root_directory)}] Found {len(image_paths)} images across {len(set(labels))} classes.")
    return image_paths, labels

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths, self.labels, self.transform = image_paths, labels, transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        path, label = self.image_paths[index], self.labels[index]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as error:
            print(f"Warning: Could not load image {path}. Skipping. Error: {error}")
            return torch.zeros((3, 224, 224)), label
        return image, label

def _assert_disjoint(name_a, set_a, name_b, set_b):
    overlap = set_a & set_b
    if overlap:
        few = list(overlap)[:5]
        raise AssertionError(f"[DATA LEAK] {name_a} and {name_b} overlap on {len(overlap)} items, e.g. {few}")

def _leak_report(train_indices_per_client=None, val_indices=None, test_indices=None, label="(indices)"):
    print(f"\n=== Data Leak Report {label} ===")
    if train_indices_per_client:
        union = set().union(*[set(s) for s in train_indices_per_client])
        total = sum(len(s) for s in train_indices_per_client)
        print(f"Clients: {len(train_indices_per_client)} shards | sum sizes={total} | union size={len(union)}")
        for i in range(len(train_indices_per_client)):
            for j in range(i+1, len(train_indices_per_client)):
                ov = set(train_indices_per_client[i]) & set(train_indices_per_client[j])
                if ov: print(f"  ⚠️ Client {i+1} ∩ Client {j+1} = {len(ov)}")
    if val_indices is not None:  print(f"Server-VAL size={len(val_indices)}")
    if test_indices is not None: print(f"TEST size={len(test_indices)}")
    print("-------------------------------------------------------------------")

def _split_server_val_from_train(train_idx, val_frac=0.1, random_state=42, stratify_y=None):
    if stratify_y is None:
        tr_idx, sv_idx = train_test_split(np.array(train_idx), test_size=val_frac, random_state=random_state, shuffle=True)
    else:
        tr_idx, sv_idx = train_test_split(
            np.array(train_idx), test_size=val_frac, random_state=random_state, shuffle=True,
            stratify=np.asarray(stratify_y)[train_idx]
        )
    return tr_idx.tolist(), sv_idx.tolist()

def _make_client_splits_indices(y, n_clients, stratified=True, random_state=42):
    y = np.asarray(y); N = len(y); all_idx = np.arange(N)
    if stratified:
        counts = np.bincount(y, minlength=int(y.max())+1)
        if (counts[counts > 0].min() >= n_clients):
            skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=random_state)
            idx_splits = [test_idx for _, test_idx in skf.split(np.zeros(N), y)]
        else:
            print(f"⚠️ Not enough samples per class for n_splits={n_clients}; using random chunks.")
            rng = np.random.RandomState(random_state)
            idx_splits = np.array_split(rng.permutation(all_idx), n_clients)
    else:
        idx_splits = np.array_split(all_idx, n_clients)
    return [np.asarray(s, dtype=int).tolist() for s in idx_splits]

# ---- Builders (DL / Traditional) ----
def build_fed_loaders_dl(
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels,
    train_tf, val_tf, batch_size, num_clients, stratified=True, random_state=42
):
    # 1) Make client shards directly from TRAIN (no peel)
    idx_splits = _make_client_splits_indices(
        train_labels, n_clients=num_clients, stratified=stratified, random_state=random_state
    )

    # >>> COMMENT OUT THIS WHOLE BLOCK (the extra peel) <<<
    # flat_train_idx = np.concatenate([np.array(s) for s in idx_splits]).tolist()
    # flat_train_idx, server_val_idx = _split_server_val_from_train(
    #     flat_train_idx, val_frac=0.1, random_state=random_state, stratify_y=np.asarray(train_labels)
    # )
    # flat_train_idx = np.array(flat_train_idx)
    # shard_sizes = [len(s) for s in idx_splits]
    # offsets = np.cumsum([0] + shard_sizes[:-1])
    # new_idx_splits = [flat_train_idx[o:o+s].tolist() for o, s in zip(offsets, shard_sizes) if s > 0]

    # Use the original idx_splits as-is
    new_idx_splits = [np.asarray(s, dtype=int).tolist() for s in idx_splits]

    # 2) Build client TRAIN loaders from the shards
    client_dataloaders = []
    for split in new_idx_splits:
        ps = [train_paths[i] for i in split]
        ls = [train_labels[i] for i in split]
        dl = DataLoader(ImageDataset(ps, ls, transform=train_tf),
                        batch_size=batch_size, shuffle=True, num_workers=0)
        client_dataloaders.append({'train': dl})

    # 3) Server-held-out VAL **uses the provided val_paths/val_labels**
    sv_paths, sv_labels = val_paths, val_labels
    server_val_loader = DataLoader(ImageDataset(sv_paths, sv_labels, transform=val_tf),
                                   batch_size=batch_size, shuffle=False, num_workers=0)

    # 4) TEST loader
    test_loader = DataLoader(ImageDataset(test_paths, test_labels, transform=val_tf),
                             batch_size=batch_size, shuffle=False, num_workers=0)

    # 5) Leak checks
    client_sets = [set(train_paths[i] for i in split) for split in new_idx_splits]
    sv_set, test_set = set(sv_paths), set(test_paths)
    for i in range(len(client_sets)):
        for j in range(i+1, len(client_sets)):
            _assert_disjoint(f"Client {i+1}", client_sets[i], f"Client {j+1}", client_sets[j])
        _assert_disjoint(f"Client {i+1}", client_sets[i], "Server-VAL", sv_set)
        _assert_disjoint(f"Client {i+1}", client_sets[i], "TEST", test_set)
    _assert_disjoint("Server-VAL", sv_set, "TEST", test_set)

    # 6) Report
    _leak_report(
        train_indices_per_client=new_idx_splits,
        val_indices=list(range(len(sv_paths))),
        test_indices=list(range(len(test_paths))),
        label="(DL indices: TRAIN shards / SERVER-VAL / TEST)"
    )

    index_ledger = {"client_indices": new_idx_splits, "server_val_indices": "provided_val"}
    return client_dataloaders, server_val_loader, test_loader, index_ledger

def build_fed_loaders_trad(X_train, y_train, X_test, y_test, batch_size, num_clients, stratified=True, random_state=42):
    idx_splits = _make_client_splits_indices(y_train, n_clients=num_clients, stratified=stratified, random_state=random_state)

    flat_train_idx = np.concatenate([np.array(s) for s in idx_splits]).tolist()
    flat_train_idx, server_val_idx = _split_server_val_from_train(
        flat_train_idx, val_frac=0.1, random_state=random_state, stratify_y=np.asarray(y_train)
    )

    flat_train_idx = np.array(flat_train_idx)
    shard_sizes = [len(s) for s in idx_splits]
    offsets = np.cumsum([0] + shard_sizes[:-1])
    new_idx_splits = [flat_train_idx[o:o+s].tolist() for o, s in zip(offsets, shard_sizes) if s > 0]

    client_dataloaders = []
    for split in new_idx_splits:
        Xi = np.asarray(X_train)[split]; yi = np.asarray(y_train)[split]
        dl = DataLoader(FeatureDataset(Xi, yi), batch_size=batch_size, shuffle=True, num_workers=0)
        client_dataloaders.append({'train': dl})

    X_sv = np.asarray(X_train)[server_val_idx]; y_sv = np.asarray(y_train)[server_val_idx]
    server_val = (X_sv, y_sv)

    for i in range(len(new_idx_splits)):
        for j in range(i+1, len(new_idx_splits)):
            _assert_disjoint(f"Client {i+1}", set(new_idx_splits[i]), f"Client {j+1}", set(new_idx_splits[j]))
        _assert_disjoint(f"Client {i+1}", set(new_idx_splits[i]), "Server-VAL", set(server_val_idx))

    _leak_report(new_idx_splits, server_val_idx, np.arange(len(X_test)), label="(Traditional indices relative to TRAIN/TEST)")

    return client_dataloaders, server_val

def coerce_features_to_float2d(X):
    arr = np.asarray(X, dtype=object)
    if arr.dtype == object:
        try: arr = np.stack(arr, axis=0)
        except Exception: arr = np.array([np.asarray(x).ravel() for x in arr], dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    return arr

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        features = coerce_features_to_float2d(features)
        labels   = np.asarray(labels, dtype=np.int64)
        self.features = torch.from_numpy(features)
        self.labels   = torch.from_numpy(labels)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def tensors_to_flat_numpy(loader, device, normalize=True):
    """
    Convert a DataLoader of (images, labels) into flat pixel arrays.
    X: (N, C*H*W) float32, y: (N,) int64
    """
    X_rows, y_rows = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Flattening to numpy", leave=False):
            # imgs: [B,C,H,W], typically already in [0,1] after ToTensor()
            imgs = imgs.to(device, non_blocking=True).float()
            if normalize and imgs.max() > 1.0:
                imgs = imgs / 255.0
            B, C, H, W = imgs.shape
            X_rows.append(imgs.view(B, -1).cpu().numpy().astype(np.float32))

            if isinstance(labels, torch.Tensor):
                y_rows.append(labels.cpu().numpy())
            else:
                y_rows.append(np.asarray(labels))
    X = np.concatenate(X_rows, axis=0)
    y = np.concatenate(y_rows, axis=0).astype(np.int64)
    return X, y