import torch #type: ignore
import numpy as np #type: ignore
import copy
from contextlib import nullcontext
from tqdm.auto import tqdm #type: ignore
from models import make_dl_model
import random
from torch.optim import AdamW #type: ignore
import torch.nn as nn #type: ignore
from sklearn.base import clone #type: ignore
from sklearn.metrics import confusion_matrix #type: ignore
from client import epoch_train, _local_train_fedprox # Import from client.py!

# Also add the missing functions: _softmax_rows, _clone_state_dict, _evaluate_server
def average_weights(w, datalens):
    total_samples = sum(datalens)
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if w_avg[key].dtype in [torch.float32, torch.float64, torch.float16]:
            w_avg[key] = torch.zeros_like(w[0][key], dtype=torch.float32)
            for i in range(len(w)):
                w_avg[key] += w[i][key] * (datalens[i] / total_samples)
        else:
            w_avg[key] = w[0][key]
    return w_avg

def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax."""
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.clip(expZ.sum(axis=1, keepdims=True), 1e-12, None)

def _clone_state_dict(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

@torch.no_grad()
def _evaluate_server(model, loader, criterion, device, use_amp=True):
    """Server-side eval on server_val/test; returns (loss, acc)."""
    is_cuda = (device.type == "cuda")
    amp_ctx = torch.amp.autocast("cuda") if (use_amp and is_cuda) else nullcontext()
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=is_cuda)
        yb = yb.to(device, non_blocking=is_cuda)
        with amp_ctx:
            logits = model(xb)
            loss = criterion(logits, yb)
        bs = yb.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(1) == yb).sum().item()
        total += bs
    loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return loss, acc

@torch.no_grad()
def epoch_eval(model, loader, criterion, device, use_amp: bool = True):
    """
    One evaluation epoch.
    - Uses torch.amp.autocast('cuda') on NVIDIA GPUs when enabled.
    - Full precision on MPS/CPU.
    Returns: (mean_loss, accuracy)
    """
    model.eval()

    is_cuda = (device.type == "cuda")
    amp_enabled = (use_amp and is_cuda)

    running_loss, running_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Validating", leave=False)

    # Choose the correct context manager
    # This is also corrected:
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        with amp_ctx:
            logits = model(inputs)
            loss = criterion(logits, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_correct += (logits.argmax(1) == labels).sum().item()
        total += bs

        pbar.set_postfix(
            loss=f"{running_loss / max(total, 1):.4f}",
            acc=f"{running_correct / max(total, 1):.4f}"
        )

    return running_loss / max(total, 1), running_correct / max(total, 1)

def epoch_eval_fed(model, loader, criterion, device, use_amp: bool = True):
    model.eval()
    is_cuda = (device.type == "cuda")
    amp_enabled = (use_amp and is_cuda)
    running_loss, running_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Validating", leave=False)
    amp_ctx = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
    all_probs, all_true = [], []
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=is_cuda)
            labels = labels.to(device, non_blocking=is_cuda)
            with amp_ctx:
                logits = model(inputs)
                loss = criterion(logits, labels)
            all_probs.append(logits.softmax(dim=1).cpu().numpy())
            all_true.append(labels.cpu().numpy())
            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(1) == labels).sum().item()
            total += bs
            pbar.set_postfix(loss=f"{running_loss/max(total,1):.4f}", acc=f"{running_correct/max(total,1):.4f}")
    return running_loss/max(total,1), running_correct/max(total,1), np.concatenate(all_true), np.concatenate(all_probs)

def federated_averaging_training(
    model_name, source_name, client_dataloaders, server_val_loader, test_loader,
    num_classes, device, rounds=10, local_epochs=5, clients_per_round=4, lr=1e-3
):
    print(f"\n{'='*20} Starting Federated Training for {model_name} {'='*20}")
    global_model = make_dl_model(model_name=model_name, num_classes=num_classes, device=device, pretrained=True)

    global_history = {'loss': [], 'accuracy': []}
    client_histories = {i: {'loss': [], 'accuracy': []} for i in range(len(client_dataloaders))}

    # NEW: remember the last local weights for each client (None if never selected)
    last_local_state = [None] * len(client_dataloaders)

    for comm_round in range(rounds):
        print(f"\n--- Communication Round {comm_round + 1}/{rounds} ---")

        local_weights, datalens = [], []
        base_state = copy.deepcopy(global_model.state_dict())

        num_to_select = min(clients_per_round, len(client_dataloaders))
        selected = random.sample(range(len(client_dataloaders)), num_to_select)
        print(f"  Selected clients this round: {[i+1 for i in selected]}")

        for ci in selected:
            loaders = client_dataloaders[ci]
            datalens.append(len(loaders['train'].dataset))

            local_model = make_dl_model(model_name=model_name, num_classes=num_classes, device=device, pretrained=True)
            local_model.load_state_dict(base_state)

            optimizer = AdamW(local_model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            print(f"  -> Training on client {ci+1} ({datalens[-1]} samples)")
            for _ in range(local_epochs):
                epoch_train(local_model, loaders['train'], optimizer, criterion, device)

            # keep local weights for aggregation + remember them for the client
            w = copy.deepcopy(local_model.state_dict())
            local_weights.append(w)
            last_local_state[ci] = w  # <-- remember

            # log client perf on server-val
            cl_loss, cl_acc, _, _ = epoch_eval_fed(local_model, server_val_loader, criterion, device)
            client_histories[ci]['loss'].append(cl_loss)
            client_histories[ci]['accuracy'].append(cl_acc)

        # aggregate
        if local_weights:
            new_global = average_weights(local_weights, datalens)
            global_model.load_state_dict(new_global)

        # server tracks on server-val
        g_loss, g_acc, _, _ = epoch_eval_fed(global_model, server_val_loader, nn.CrossEntropyLoss(), device)
        global_history['loss'].append(g_loss)
        global_history['accuracy'].append(g_acc)
        print(f"Round {comm_round+1} | Server (on server-val) acc: {g_acc*100:.2f}%")

    final_client_cms = {}
    for ci, loaders in enumerate(client_dataloaders):
        cm_loss, cm_acc, cm, _ = epoch_eval_fed(
            global_model, loaders['train'], nn.CrossEntropyLoss(), device
        )
        final_client_cms[ci] = cm

        # ---- NEW: Build client_pairs for panel plots / per-client ROC ----
    client_pairs = []
    criterion = nn.CrossEntropyLoss()
    for ci, loaders in enumerate(client_dataloaders):
        eval_loader = loaders.get("test", None) or test_loader
        _, _, y_true_c, probs_c = epoch_eval_fed(global_model, eval_loader, criterion, device)
        client_pairs.append((y_true_c, probs_c))

    # ---- Return EXACTLY what main.py expects to unpack ----
    return global_model, global_history, client_histories, final_client_cms, last_local_state, client_pairs


def federated_training_traditional_ensemble(model_template, client_dataloaders, server_val):
    print(f"Starting federated training (ensemble) for {type(model_template).__name__}...")
    X_sv, y_sv = server_val
    client_models, client_cms = [], []
    for i, loader_d in enumerate(client_dataloaders):
        print(f"  Training client {i+1}/{len(client_dataloaders)}...")
        m = clone(model_template)
        Xc, yc = [], []
        for Xb, yb in loader_d['train']:
            Xc.append(Xb.numpy()); yc.append(yb.numpy())
        if not Xc: continue
        Xc = np.concatenate(Xc); yc = np.concatenate(yc)
        m.fit(Xc, yc); client_models.append(m)
        y_pred_val = m.predict(X_sv)
        cm_val = confusion_matrix(y_sv, y_pred_val, labels=np.unique(y_sv))
        client_cms.append(cm_val)
    print("Federated ensemble training complete.")
    return client_models, client_cms

def federated_training_traditional_linear(model_template, client_dataloaders, rounds=10):
    print(f"Starting federated training (linear) for {type(model_template).__name__}...")
    global_model = clone(model_template)
    X_sample, y_sample = next(iter(client_dataloaders[0]['train']))
    global_model.fit(X_sample.numpy(), y_sample.numpy())
    for r in range(rounds):
        clients = []
        print(f"  Round {r+1}/{rounds}...")
        for loader_d in client_dataloaders:
            m = clone(global_model)
            Xc, yc = [], []
            for Xb, yb in loader_d['train']:
                Xc.append(Xb.numpy()); yc.append(yb.numpy())
            if not Xc: continue
            Xc = np.concatenate(Xc); yc = np.concatenate(yc)
            m.fit(Xc, yc); clients.append(m)
        if clients:
            global_model.coef_      = np.mean([m.coef_ for m in clients], axis=0)
            global_model.intercept_ = np.mean([m.intercept_ for m in clients], axis=0)
    print("Federated training for linear model complete.")
    return global_model

def federated_prox_training(
    model_name: str,
    source_name: str,
    client_dataloaders: list,           # list of dicts with 'train' (and optionally 'val')
    server_val_loader,                  # DataLoader
    test_loader,                        # DataLoader
    num_classes: int,
    device: torch.device,
    rounds: int = 10,
    local_epochs: int = 1,
    clients_per_round: int = None,
    lr: float = 1e-3,
    mu: float = 0.001,                  # FedProx strength
    make_model_fn=None,                 # factory: (model_name, num_classes) -> nn.Module
    build_model_fn=None,                # optional alternative factory signature
    criterion=None                      # if None -> CrossEntropyLoss
):
    """
    FedProx trainer: returns (global_model, global_history, client_histories, final_client_cms)
      - global_history: {'loss': [...], 'accuracy': [...]}
      - client_histories: {client_idx: {'loss': [...], 'accuracy': [...]} }
      - final_client_cms: list of per-client CMs (all clients; fallback to global if never selected)
    """
    assert len(client_dataloaders) > 0, "No client dataloaders provided."

    # ---- Build/initialize the global model
    if make_model_fn is None and 'make_dl_model' in globals():
        make_model_fn = lambda name, nc: make_dl_model(model_name=name, num_classes=nc, device=device, pretrained=True)
    if make_model_fn is not None:
        global_model = make_model_fn(model_name, num_classes)
    elif build_model_fn is not None:
        global_model = build_model_fn(model_name=model_name, num_classes=num_classes)
    else:
        raise ValueError("Provide make_model_fn or build_model_fn to create the model.")
    global_model.to(device)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    rng = np.random.RandomState(42)
    n_clients_total = len(client_dataloaders)
    if not clients_per_round or clients_per_round <= 0:
        clients_per_round = n_clients_total
    clients_per_round = min(clients_per_round, n_clients_total)

    # Histories
    global_history = {"loss": [], "accuracy": []}
    client_histories = {i: {"loss": [], "accuracy": []} for i in range(n_clients_total)}
    last_local_state = [None] * n_clients_total   # track per-client last local model

    for r in range(1, rounds + 1):
        print(f"\n[Round {r}/{rounds}] FedProx (mu={mu})")

        # Select clients (deterministic per rng above)
        selected = rng.choice(n_clients_total, size=clients_per_round, replace=False)
        selected = sorted(selected.tolist())
        print(f"  Selected clients this round: {[i+1 for i in selected]}")

        # Cache global weights
        global_params = _clone_state_dict(global_model)

        local_weights, local_sizes = [], []

        # Train each selected client
        for ci in selected:
            client_model = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)

            train_loader = client_dataloaders[ci]['train']
            sz = len(train_loader.dataset)

            loss_c, acc_c = _local_train_fedprox(
                client_model, train_loader, criterion, optimizer, device,
                global_params=global_params, mu=mu, local_epochs=local_epochs
            )

            # record per-client history for this round
            client_histories[ci]['loss'].append(loss_c)
            client_histories[ci]['accuracy'].append(acc_c)

            # collect for aggregation + remember this client's last state
            w = _clone_state_dict(client_model)
            local_weights.append(w); local_sizes.append(sz)
            last_local_state[ci] = w

        # Aggregate to form new global
        if local_weights:
            w_avg = average_weights(local_weights, local_sizes)
            global_model.load_state_dict(w_avg, strict=True)

        # Server validation for this round
        sv_loss, sv_acc = _evaluate_server(global_model, server_val_loader, criterion, device)
        global_history['loss'].append(sv_loss)
        global_history['accuracy'].append(sv_acc)
        print(f"  [Server] val_loss={sv_loss:.4f}  val_acc={sv_acc:.4f}")

    # ---------- FINAL PER-CLIENT CMs ON TEST SET ----------
    print("\n--- Generating final confusion matrices from *each client's last local model* (FedProx) ---")
    final_client_cms = []
    for ci in range(n_clients_total):
        cm_model = copy.deepcopy(global_model).to(device)
        if last_local_state[ci] is not None:
            cm_model.load_state_dict(last_local_state[ci])
        cm_model.eval()
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = cm_model(xb.to(device))
                y_pred_all.append(logits.argmax(1).cpu().numpy())
                y_true_all.append(yb.numpy())
        y_true_client = np.concatenate(y_true_all)
        y_pred_client = np.concatenate(y_pred_all)
        cm = confusion_matrix(y_true_client, y_pred_client, labels=list(range(num_classes)))
        final_client_cms.append(cm)

    return global_model, global_history, client_histories, final_client_cms

def evaluate_federated_ensemble(
    client_models,
    X_test,
    y_test=None,
    class_labels=None,
    explain=True,
    k=3,
    weights=None,
    **kwargs,   # <-- accepts legacy names without breaking
):
    """
    Server-side aggregation: weighted average of client probability matrices with class alignment.
    Backward compatible with legacy params:
      - 'show_k' (alias of k)
      - 'verbose' (alias of explain)
    Returns:
      y_pred: (N,) int
      avg_P : (N, C) float, aligned to class_labels (if provided) or union of client classes
    """
    # ---- Backwards-compat for old call sites ----
    if "show_k" in kwargs and kwargs["show_k"] is not None:
        k = kwargs["show_k"]
    if "verbose" in kwargs and kwargs["verbose"] is not None:
        explain = kwargs["verbose"]

    if not client_models:
        return None, None

    # 1) Collect raw (P_i, classes_i) and union of classes
    raw = [_get_proba_and_classes(m, X_test) for m in client_models]  # list of (P_i, classes_i)
    class_union = None
    for _, classes_i in raw:
        class_union = classes_i if class_union is None else np.unique(np.concatenate([class_union, classes_i]))

    # If canonical order is known (class_labels), prefer 0..C-1 based on that
    if class_labels is not None and len(class_labels) > 0:
        # assume labels map to indices 0..C-1; enforce that ordering
        class_union = np.arange(len(class_labels))

    # 2) Align probabilities to the union order
    probs = [_align_probs_to_classes(P, classes_i, class_union) for (P, classes_i) in raw]
    K = len(probs); N, C = probs[0].shape
    assert all(p.shape == (N, C) for p in probs), "Aligned prob shapes mismatch."

    # 3) Weights (default uniform), normalize safely
    if weights is None:
        w = np.full(K, 1.0 / K, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != (K,):
            raise ValueError(f"weights must have shape {(K,)}, got {w.shape}")
        s = w.sum()
        w = w / (s if s != 0 else 1.0)

    # 4) Weighted average across clients → (N, C)
    stack = np.stack(probs, axis=0)            # (K, N, C)
    avg_P = np.tensordot(w, stack, axes=(0, 0))  # (N, C)
    y_pred = np.argmax(avg_P, axis=1)

    # 5) Optional console explainer
    if explain:
        print("\n[Server aggregation]")
        print(f"  Clients: {K} | After alignment: (N={N}, C={C})")
        print(f"  Rule: avg_P = Σ_i w_i * P_i (aligned)")
        print(f"  Weights: {np.round(w, 4).tolist()}  (sum={float(w.sum()):.4f})")
        for i, (P_i, classes_i) in enumerate(raw, 1):
            print(f"  Client {i}: raw {P_i.shape}, classes={classes_i.tolist()} -> aligned {probs[i-1].shape}")

        kk = min(int(k), N)
        print(f"\n  Examples (first {kk} rows): top-1 per client → avg → pred [true]")
        for r in range(kk):
            parts = []
            for i in range(K):
                c_i = int(np.argmax(probs[i][r]))
                if class_labels is not None and c_i < len(class_labels):
                    lbl = class_labels[c_i]
                else:
                    lbl = c_i
                parts.append(f"C{i+1}:{lbl}({probs[i][r, c_i]:.3f})")
            c_avg = int(y_pred[r])
            lbl_avg = class_labels[c_avg] if (class_labels is not None and c_avg < len(class_labels)) else c_avg
            line = " | ".join(parts) + f"  → avg:{lbl_avg}({avg_P[r, c_avg]:.3f})"
            if y_test is not None:
                t = int(y_test[r])
                lbl_t = class_labels[t] if (class_labels is not None and t < len(class_labels)) else t
                line += f"  [true={lbl_t}]"
            print("   ", line)

        if y_test is not None:
            acc = float((y_pred == np.asarray(y_test)).mean())
            print(f"\n  Quick accuracy on provided y_test: {acc*100:.2f}%")

    return y_pred, avg_P

# ===== Robust probability alignment + server aggregation =====
def _get_proba_and_classes(model, X):
    """Return (P, classes) where P is (N,C_model) and classes is the label list the model knows."""
    if hasattr(model, "predict_proba"):
        P = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(model, "classes_", np.arange(P.shape[1])))
        return P, classes
    if hasattr(model, "decision_function"):
        Z = np.asarray(model.decision_function(X), dtype=float)
        if Z.ndim == 1:  # binary
            Z = np.stack([-Z, Z], axis=1)
            classes = np.asarray([0, 1])
        else:
            classes = np.arange(Z.shape[1])
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z); P /= np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)
        return P, classes
    raise AttributeError("Model must implement predict_proba or decision_function.")

def _align_probs_to_classes(P, model_classes, target_classes):
    """
    Align a client's probability matrix (N, C_model) to the union class order target_classes (C*).
    Any missing classes get probability 0.
    """
    target_index = {c: j for j, c in enumerate(target_classes)}
    N = P.shape[0]; C_star = len(target_classes)
    out = np.zeros((N, C_star), dtype=float)
    for j_model, c in enumerate(model_classes):
        j_target = target_index.get(c, None)
        if j_target is not None:
            out[:, j_target] = P[:, j_model]
    return out

