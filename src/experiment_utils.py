import os
import sys
import json
from math import ceil
from data_prep import SOURCE_CLASS_LABELS
from models import make_dl_model
import numpy as np # type: ignore
from typing import List, Tuple, Dict

from config import (RUN_MODE,
                     DATASETS, 
                    RUN_ON_DATASET,
                    RUN_ON_DATASET)
# ===================================================================
# EXPERIMENT SCAFFOLD & CONFIGURATION
# ===================================================================
def _require(name, cond, msg=None):
    if not cond:
        raise RuntimeError(msg or f"Required condition failed: {name}")

def _warn(msg):
    print(f"[warn] {msg}", file=sys.stderr)

def _exists_callable(name):
    return (name in globals()) and callable(globals()[name])

def _check_required_functions(main_globals):  # <-- Add 'main_globals'
    needed = [
        "build_fed_loaders_dl", "build_fed_loaders_trad",
        "federated_averaging_training", "federated_training_traditional_ensemble",
        "_safe_auc_multiclass", "_macro_prec_recall_from_report",
        "plot_fed_server_roc_cm",
        "get_transforms"
    ]

    # Check the passed-in 'main_globals', not the local ones
    missing = [fn for fn in needed if not (fn in main_globals and callable(main_globals[fn]))]

    _require("required functions present", len(missing) == 0,
             f"Missing functions: {missing}")

def _check_globals_present(main_globals):  # <-- Add 'main_globals'
    needed = [
        "SOURCE_CLASS_LABELS", "all_dataloaders_info", "all_numpy_data",
        "TRADITIONAL_MODELS_MAP", "DATASETS",
        "BATCH_SIZE", "DEVICE", "RANDOM_STATE", "FED_ROUNDS", "FED_LEARNING_RATE", "FED_LOCAL_EPOCHS",
        "RUN_FEDERATED_LEARNING", "RUN_MODE", "DEEP_MODELS", "TRADITIONAL_MODELS", "datasets_to_run"
    ]
    missing = [g for g in needed if g not in main_globals]
    _require("required globals present", len(missing) == 0,
             f"Missing globals: {missing}")

def _check_dataset_keys(main_globals):  # <-- Add 'main_globals'
    all_dataloaders_info = main_globals.get("all_dataloaders_info", {})
    all_numpy_data = main_globals.get("all_numpy_data", {})
    datasets_to_run = main_globals.get("datasets_to_run", {})       
    """
    Validate each source in datasets_to_run for requested modalities and build DATASETS_PLAN.
    Deep  (RUN_MODE in {"deep","both"}):  must exist in all_dataloaders_info
    Trad  (RUN_MODE == "traditional"):    must exist in all_numpy_data
          (RUN_MODE == "both"):           warn+skip trad if missing
    Class labels:                         required for any modality
    """
    want_deep = RUN_MODE in ("deep", "both")
    want_trad = RUN_MODE in ("traditional", "both")

    plan = {}

    for src in list(datasets_to_run.keys()):
        _require(f"{src} in SOURCE_CLASS_LABELS",
                 src in SOURCE_CLASS_LABELS,
                 f"Missing class labels for source '{src}' in SOURCE_CLASS_LABELS.")

        has_deep = src in all_dataloaders_info
        has_trad = src in all_numpy_data

        if want_deep:
            _require(f"{src} in all_dataloaders_info", has_deep,
                     f"Source '{src}' not found in all_dataloaders_info.")

        if want_trad and not has_trad:
            if RUN_MODE == "both":
                _warn(f"Skipping TRADITIONAL for '{src}' (no NumPy data). Will run DEEP only.")
            else:
                _require(f"{src} in all_numpy_data", False,
                         f"Source '{src}' not found in all_numpy_data.")

        plan[src] = {
            "deep": has_deep and want_deep,
            "traditional": has_trad and want_trad
        }

        if RUN_MODE == "both" and not (plan[src]["deep"] or plan[src]["traditional"]):
            _require("at least one modality available", False,
                     f"No usable data for '{src}' (missing in both loaders).")

    globals()["DATASETS_PLAN"] = plan
    print("[plan] DATASETS_PLAN:", {k: {m: int(v) for m, v in v2.items()} for k, v2 in plan.items()})
    return plan

def _check_histories(hist, name="hist"):
    ok = isinstance(hist, dict) and ("loss" in hist) and ("accuracy" in hist)
    if ok and (len(hist.get("loss", [])) == 0 or len(hist.get("accuracy", [])) == 0):
        ok = False
    if not ok:
        _warn(f"{name} missing/empty.")
    return ok

def _collect_client_test_probs_from_states(
    model_name, num_classes, device, last_local_state_list, test_loader
):
    """Returns a list of (y_true, probs) for each client state; uses global if state is None."""
    pairs = []
    for st in last_local_state_list:
        cm_model = make_dl_model(model_name=model_name, num_classes=num_classes, device=device, pretrained=True)
        if st is not None:
            cm_model.load_state_dict(st, strict=True)
        cm_model.eval()
        ys, Ps = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = cm_model(xb)
                Ps.append(logits.softmax(dim=1).cpu().numpy())
                ys.append(yb.numpy())
        y_true_c = np.concatenate(ys)
        y_prob_c = np.concatenate(Ps)
        pairs.append((y_true_c, y_prob_c))
    return pairs

def _check_histories(hist, name="hist"):
    ok = isinstance(hist, dict) and ("loss" in hist) and ("accuracy" in hist)
    if ok and (len(hist.get("loss", [])) == 0 or len(hist.get("accuracy", [])) == 0):
        ok = False
    if not ok:
        _warn(f"{name} missing/empty.")
    return ok

# --- Fallback SAVE_DIR if not set (assumes SAVE_DIR is from global config cell)
if 'SAVE_DIR' not in globals():
    SAVE_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(SAVE_DIR, exist_ok=True)
EXP_BASE_DIR = os.path.join(SAVE_DIR, "experiments")
os.makedirs(EXP_BASE_DIR, exist_ok=True)

# --- Distribution names & definitions
DIST_INDEPENDENT = "Independent and Identically Distributed"
DIST_LABEL_SKEW  = "Non-Identically Distributed (Label Skew)"
DIST_QTY_SKEW    = "Non-Identically Distributed (Quantity Skew)"

DIST_DEFS = {
    DIST_INDEPENDENT: (
        "Each client has data randomly sampled from the same overall distribution. "
        "Example: if the global dataset has 10 classes, each client has a balanced mix of all 10 classes. "
    ),
    DIST_LABEL_SKEW: (
        "Each client has data from only a subset of labels. "
        "Example: one client has only classes 0–2, another has only 3–5. "
    ),
    DIST_QTY_SKEW: (
        "All clients see all labels, but the amount of data per client varies. "
        "Example: one client has 500 samples; another has only 50. "
    ),
}

# --- Dataset selection (assumes RUN_ON_DATASET and DATASETS are global)
if RUN_ON_DATASET == "Both":
    datasets_to_run = DATASETS
    print("🚀 Running on BOTH 'Original' and 'Geometric' datasets.")
elif RUN_ON_DATASET in DATASETS:
    datasets_to_run = {RUN_ON_DATASET: DATASETS[RUN_ON_DATASET]}
    print(f"🚀 Running ONLY on the '{RUN_ON_DATASET}' dataset.")
else:
    datasets_to_run = {}
    print(f"⚠️ Warning: Dataset '{RUN_ON_DATASET}' not found. No datasets will be processed.")
print("Datasets to run:", list(datasets_to_run.keys()))

# --- Experiment loop helpers
def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in str(s)).strip("_").replace("__","_")

def _exp_dir(exp_id, source_name, n_clients, distribution, participation, method):
    dist_slug = _slug(distribution.replace("(", "").replace(")", ""))
    src_slug  = _slug(source_name)
    method_s  = _slug(method)
    tag = f"exp{int(exp_id):02d}_{src_slug}_clients{n_clients}_{dist_slug}_participation{int(participation*100)}_{method_s}"
    path = os.path.join(EXP_BASE_DIR, tag)
    os.makedirs(path, exist_ok=True)
    return path

def _approx_split_flags(distribution: str) -> bool:
    return "independent and identically distributed" in distribution.strip().lower()

def _subset_clients_for_participation(client_dls, participation_rate: float, seed: int = 42):
    if participation_rate >= 0.999:
        return client_dls
    rng = np.random.RandomState(seed)
    k = max(1, int(ceil(participation_rate * len(client_dls))))
    idx = rng.choice(len(client_dls), size=k, replace=False)
    return [client_dls[i] for i in sorted(idx.tolist())]

def _collect_class_counts_from_dl(dl, class_names=None):
    lab = dl['train'].dataset.labels
    ys = lab.cpu().numpy() if hasattr(lab, "cpu") else (lab if isinstance(lab, np.ndarray) else np.asarray(lab))
    uniq, cnt = np.unique(ys, return_counts=True)
    if class_names and int(np.max(uniq)) < len(class_names):
        name_map = {k: class_names[k] for k in range(len(class_names))}
        return {name_map[int(k)]: int(v) for k, v in zip(uniq, cnt)}
    return {int(k): int(v) for k, v in zip(uniq, cnt)}

def _print_split_diag_DL(client_dls, server_val_loader, test_loader, source_name):
    _dl_class_names = SOURCE_CLASS_LABELS.get(source_name, [])
    sizes = [len(dl['train'].dataset) for dl in client_dls]
    total = int(np.sum(sizes))
    print("\n[Split check: Deep Learning] clients=", len(client_dls),
          "| shard_sizes=", sizes,
          f"| server_val={len(server_val_loader.dataset)}",
          f"| test={len(test_loader.dataset)}")
    for i, dl in enumerate(client_dls, start=0):
        dist = _collect_class_counts_from_dl(dl, _dl_class_names)
        print(f"  Client {i+1} class counts:", dist)

def _print_split_diag_trad(client_dls, server_val, X_test, class_labels):
    sizes = [len(dl['train'].dataset) for dl in client_dls]
    total = int(np.sum(sizes))
    print("\n[Split check: Traditional] clients=", len(client_dls),
          "| shard_sizes=", sizes,
          f"| server_val={len(server_val[1])}",
          f"| test={len(X_test)}")
    for i, dl in enumerate(client_dls, start=0):
        dist = _collect_class_counts_from_dl(dl, class_labels)
        print(f"  Client {i+1} class counts:", dist)
    _, y_sv = server_val
    u_sv, c_sv = np.unique(y_sv, return_counts=True)
    print("  Server-VAL class counts:", {class_labels[int(k)]: int(v) for k, v in zip(u_sv, c_sv)})

def _save_meta(path, payload: dict):
    try:
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        _warn(f"couldn't save meta.json to {path}: {e}")
