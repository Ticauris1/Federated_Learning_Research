# (You already have these)
import os
import csv
from experiment_utils import _approx_split_flags, _check_dataset_keys, _check_globals_present, _check_histories, _check_required_functions, _exp_dir, _print_split_diag_DL, _print_split_diag_trad, _save_meta, _subset_clients_for_participation, _warn
import numpy as np # type: ignore
import pandas as pd # type: ignore
import torch # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from torchvision import transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
from data_utils import ImageDataset # Already in data_utils
from experiment_utils import *
# --- 2. Import From Your NEW .py Files ---
import config

# Import specific functions from your data utilities file
from data_utils import (
    find_image_files, 
    build_fed_loaders_dl, 
    build_fed_loaders_trad
)
from config import (PIXEL_IMG_SIZE, RUN_MODE, DATASETS, RUN_ON_DATASET,
                    DEVICE, FED_LEARNING_RATE, FED_LOCAL_EPOCHS,
                    RUN_FEDERATED_LEARNING, RUN_ON_DATASET, DEEP_MODELS, 
                    TRADITIONAL_MODELS, TRADITIONAL_MODELS_MAP, FED_ROUNDS,
                    FED_LOCAL_EPOCHS, FED_LEARNING_RATE, BATCH_SIZE, RANDOM_STATE)

# Import the model factory function
from models import get_transforms, make_dl_model

# Import your main server-side training loops
from server import (
    federated_averaging_training, 
    federated_training_traditional_ensemble, 
    evaluate_federated_ensemble
)

# Import your traditional ML pipeline and helper
from data_utils import tensors_to_flat_numpy # (You used this in data prep)

# Import all your plotting functions
from plotting import (
    plot_server_with_clients_one_axes,
    plot_col_cms,
    plot_col_rocs,
    plot_grid_cms,
    plot_grid_rocs,
    plot_fed_server_roc_cm
)

# Import helper functions
from federated_utils import (
    _safe_auc_multiclass, 
    _macro_prec_recall_from_report,
    _client_test_probs_trad
)


from data_utils import (
    find_image_files
)

# ============================================================
# DATASET PREPARATION
# (Relies on user-defined helpers: find_image_files, tensors_to_flat_numpy,
#  ImageDataset, BATCH_SIZE, DEVICE, RANDOM_STATE, LabelEncoder, transforms, etc.)
# ============================================================

if 'all_results_rows' not in globals():
    all_results_rows, all_confusion_matrices, all_probs, all_histories = [], {}, {}, {}
if 'all_dataloaders_info' not in globals():
    all_dataloaders_info, all_numpy_data, SOURCE_CLASS_LABELS = {}, {}, {}

# build datasets_to_run from RUN_ON_DATASET / DATASETS
if RUN_ON_DATASET == "Both":
    datasets_to_run = DATASETS
elif RUN_ON_DATASET in DATASETS:
    datasets_to_run = {RUN_ON_DATASET: DATASETS[RUN_ON_DATASET]}
else:
    datasets_to_run = {}

for source_name, source_path in datasets_to_run.items():
    print(f"\n{'='*28} PREPARING SOURCE: {source_name} {'='*28}")
    image_paths, string_labels = find_image_files(source_path)
    if not image_paths:
        print(f"⚠️ No images found for {source_name}; skipping.")
        continue

    le = LabelEncoder()
    int_labels = le.fit_transform(string_labels)
    class_labels = list(le.classes_)
    num_classes = len(class_labels)
    SOURCE_CLASS_LABELS[source_name] = class_labels
    print(f"[{source_name}] Classes: {class_labels}")

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, int_labels, test_size=0.15, random_state=RANDOM_STATE, stratify=int_labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.15/0.85, random_state=RANDOM_STATE, stratify=train_val_labels
    )

    all_dataloaders_info[source_name] = dict(
        train_paths=train_paths, train_labels=train_labels,
        val_paths=val_paths,     val_labels=val_labels,
        test_paths=test_paths,   test_labels=test_labels,
        num_classes=num_classes, class_labels=class_labels
    )

    pixel_tf = transforms.Compose([transforms.Resize((PIXEL_IMG_SIZE, PIXEL_IMG_SIZE)), transforms.ToTensor()])
    X_train_px, y_train_px = tensors_to_flat_numpy(
        DataLoader(ImageDataset(train_paths, train_labels, transform=pixel_tf), batch_size=BATCH_SIZE, num_workers=0), DEVICE)
    X_test_px,  y_test_px  = tensors_to_flat_numpy(
        DataLoader(ImageDataset(test_paths, test_labels, transform=pixel_tf), batch_size=BATCH_SIZE, num_workers=0),  DEVICE)
    all_numpy_data[source_name] = {'X_train': X_train_px, 'y_train': y_train_px, 'X_test': X_test_px, 'y_test': y_test_px}

print("Prep summary:")
print("  numpy_data keys:     ", sorted(all_numpy_data.keys()))
print("  dl_info keys:        ", sorted(all_dataloaders_info.keys()))

# --- Sanity checks (Assumes these functions are defined elsewhere)
print("RUN_MODE:", RUN_MODE)
print("all_numpy_data keys:", list(all_numpy_data.keys()))
print("all_dataloaders_info keys:", list(all_dataloaders_info.keys()))

if RUN_MODE == "traditional" and len(all_numpy_data) == 0:
    raise RuntimeError("RUN_MODE='traditional' but all_numpy_data is empty.")
if RUN_MODE == "both" and len(all_numpy_data) == 0:
    _warn("all_numpy_data is empty. Traditional path will be skipped.")

_check_required_functions(globals())
_check_globals_present(globals())
_check_dataset_keys(globals())

# --- Experiment Grid Definition
EXPERIMENTS = [
    {"id":  1, "clients": 2, "dist": DIST_INDEPENDENT, "pr": 1.00, "method": "FedAvg"},
    {"id":  2, "clients": 3, "dist": DIST_INDEPENDENT, "pr": 1.00, "method": "FedAvg"},
    {"id":  3, "clients": 4, "dist": DIST_INDEPENDENT, "pr": 1.00, "method": "FedAvg"},
]

# --- Summary CSV Setup
SUMMARY_CSV = os.path.join(EXP_BASE_DIR, "_summary.csv")

_summary_fields = ["experiment","source","model_name","clients","distribution","participation","method",
                   "accuracy","precision","recall","f1_score","auc_score","save_dir"]
if not os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_summary_fields).writeheader()

print("✅ Experiment scaffold and configuration complete.")

# ===================================================================
# MAIN EXPERIMENT LOOP
# ===================================================================

# --- Start Loop ---
for exp in EXPERIMENTS:
    exp_id    = exp["id"]
    n_clients = exp["clients"]
    dist      = exp["dist"]
    prate     = float(exp["pr"])
    method    = exp["method"]

    print("\n" + "="*110)
    print(f"Experiment #{exp_id} | Clients={n_clients} | Distribution={dist} | Participation={int(prate*100)}% | Method={method}")
    print("="*110)

    stratified_flag = _approx_split_flags(dist)

    # ==================================
    # ---- DEEP LEARNING PATH ----
    # ==================================
    if RUN_FEDERATED_LEARNING and RUN_MODE in ["deep", "both"] and len(all_dataloaders_info):
        num_classes_fed = next(iter(all_dataloaders_info.values()))['num_classes']
        for model_name in DEEP_MODELS:
            for source_name, _ in datasets_to_run.items():
                
                # ---- MODIFICATION START ----
                # 1. Get the base experiment directory
                base_exp_dir = _exp_dir(exp_id, source_name, n_clients, dist, prate, method)
                # 2. Create the new model-specific subdirectory
                exp_dir = os.path.join(base_exp_dir, model_name)
                # 3. Ensure this new directory exists
                os.makedirs(exp_dir, exist_ok=True)
                # 4. Update the print statement to show the new path
                print(f"[Experiment save dir] {exp_dir}")
                # ---- MODIFICATION END ----

                # Client metrics CSV (per-experiment)
                CLIENT_SUMMARY_CSV = os.path.join(exp_dir, "_client_summary.csv")
                _client_summary_fields = [
                    "experiment", "source", "model_name", "client_id",
                    "accuracy", "precision", "recall", "f1_score", "auc_score"
                ]
                with open(CLIENT_SUMMARY_CSV, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=_client_summary_fields).writeheader()

                # Build loaders
                data = all_dataloaders_info[source_name]
                train_tf, val_tf = get_transforms(model_name)
                client_dls, server_val_loader, test_loader, _ = build_fed_loaders_dl(
                    data['train_paths'], data['train_labels'],
                    data['val_paths'],   data['val_labels'],
                    data['test_paths'],  data['test_labels'],
                    train_tf=train_tf, val_tf=val_tf, batch_size=BATCH_SIZE,
                    num_clients=n_clients, stratified=stratified_flag, random_state=RANDOM_STATE
                )
                _print_split_diag_DL(client_dls, server_val_loader, test_loader, source_name)
                _save_meta(exp_dir, {
                    "clients": n_clients, "distribution": dist, "distribution_definition": DIST_DEFS.get(dist, ""),
                    "participation": prate, "method": method,
                    "client_shard_sizes": [len(dl['train'].dataset) for dl in client_dls],
                    "server_val_size": len(server_val_loader.dataset), "test_size": len(test_loader.dataset),
                })

                cpr = max(1, int(ceil(prate * n_clients)))
                if method == "FedProx":
                    print("  [note] FedProx not implemented; running FedAvg baseline.")

                # ===== Train Federated =====
                fed_model, fed_history, client_histories, final_client_cms, last_local_state = federated_averaging_training(
                    model_name=model_name, source_name=source_name,
                    client_dataloaders=client_dls,
                    server_val_loader=server_val_loader,
                    test_loader=test_loader,
                    num_classes=num_classes_fed, device=DEVICE,
                    rounds=FED_ROUNDS, local_epochs=FED_LOCAL_EPOCHS,
                    clients_per_round=cpr, lr=FED_LEARNING_RATE
                )

                # --- Curve overlays ---
                try:
                    if _check_histories(fed_history, "fed_history"):
                        plot_server_with_clients_one_axes(
                            global_history=fed_history, client_histories=client_histories,
                            save_dir=exp_dir, model_name=model_name,
                            source_name=source_name, metric="loss"
                        )
                        plot_server_with_clients_one_axes(
                            global_history=fed_history, client_histories=client_histories,
                            save_dir=exp_dir, model_name=model_name,
                            source_name=source_name, metric="accuracy"
                        )
                except Exception as e:
                    _warn(f"Curve overlay plot error: {e}")

                # --- Evaluate server model ---
                class_labels = SOURCE_CLASS_LABELS[source_name]
                num_classes = len(class_labels)
                fed_model.eval()
                probs, y_true = [], []
                with torch.no_grad():
                    for xb, yb in test_loader:
                        probs.append(fed_model(xb.to(DEVICE)).softmax(dim=1).cpu().numpy())
                        y_true.append(yb.numpy())
                probs, y_true = np.concatenate(probs), np.concatenate(y_true)
                y_pred = probs.argmax(axis=1)
                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

                # --- Get client results ---
                server_y_true, server_y_prob = y_true, probs


                # --- Plot Panels ---
                try:
                    experiment_data_dict = {
                        "title": f"{n_clients} clients",
                        "server_cm": cm,
                        "client_cms": final_client_cms,
                        "server_y_true": server_y_true,
                        "server_y_prob": server_y_prob,
                        "client_pairs": client_pairs,
                        "class_labels": class_labels,
                        "model_name": model_name,
                        "source_name": source_name,
                    }
                    # Column
                    plot_col_cms(experiment_data_dict, save_dir=exp_dir)
                    plot_col_rocs(experiment_data_dict, save_dir=exp_dir)
                    # Grid
                    plot_grid_cms(
                        cm, final_client_cms, exp_dir, model_name, source_name, # Fixed: was client_cms_test
                        class_labels=class_labels, dpi=180
                    )
                    plot_grid_rocs(
                        server_y_true, server_y_prob, client_pairs, exp_dir, model_name, source_name,
                        class_labels=class_labels, dpi=180
                    )
                except Exception as e:
                    _warn(f"Panel plotting error: {e}")

                # --- Log client metrics ---
                # NOTE: Re-evaluating clients on test set as `final_client_cms` was on local data
                client_cms_test = []
                for i, (y_true_client_test, probs_client_test) in enumerate(client_pairs):
                    if y_true_client_test is None or probs_client_test is None:
                        _warn(f"Skipping client {i+1} metrics, no test probs.")
                        client_cms_test.append(np.zeros((num_classes, num_classes))) # Append empty CM
                        continue
                        
                    y_pred_client_test = probs_client_test.argmax(axis=1)
                    client_cms_test.append(confusion_matrix(y_true_client_test, y_pred_client_test, labels=list(range(num_classes))))
                    
                    report_client = classification_report(y_true_client_test, y_pred_client_test, target_names=class_labels, output_dict=True, zero_division=0)
                    prec_client, rec_client, f1_client = _macro_prec_recall_from_report(report_client)
                    auc_client = _safe_auc_multiclass(y_true_client_test, probs_client_test, num_classes)
                    with open(CLIENT_SUMMARY_CSV, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=_client_summary_fields)
                        writer.writerow({
                            "experiment": exp_id, "source": source_name, "model_name": f"{model_name}-FedAvg",
                            "client_id": i + 1, "accuracy": (y_pred_client_test == y_true_client_test).mean(),
                            "precision": prec_client, "recall": rec_client, "f1_score": f1_client, "auc_score": auc_client
                        })

                # --- Log server metrics ---
                report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
                macro_prec, macro_rec, macro_f1 = _macro_prec_recall_from_report(report)
                auc_macro = _safe_auc_multiclass(y_true, probs, num_classes)
                with open(SUMMARY_CSV, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=_summary_fields)
                    writer.writerow({
                        "experiment": exp_id, "source": source_name, "model_name": f"{model_name}-FedAvg",
                        "clients": n_clients, "distribution": dist, "participation": prate, "method": method,
                        "accuracy": (y_pred == y_true).mean(),
                        "precision": macro_prec, "recall": macro_rec, "f1_score": macro_f1,
                        "auc_score": auc_macro, "save_dir": exp_dir
                    })

    # ==================================
    # ---- TRADITIONAL ML PATH ----
    # ==================================
    if RUN_FEDERATED_LEARNING and RUN_MODE in ["traditional", "both"] and len(all_numpy_data):
        for model_name in TRADITIONAL_MODELS:
            for source_name, _ in datasets_to_run.items():
                if source_name not in all_numpy_data:
                    _warn(f"Skipping traditional model for {source_name}, no numpy data found.")
                    continue

                # ---- MODIFICATION START ----
                # 1. Get the base experiment directory
                base_exp_dir = _exp_dir(exp_id, source_name, n_clients, dist, prate, method)
                # 2. Create the new model-specific subdirectory
                exp_dir = os.path.join(base_exp_dir, model_name)
                # 3. Ensure this new directory exists
                os.makedirs(exp_dir, exist_ok=True)
                # 4. Update the print statement to show the new path
                print(f"[Experiment save dir] {exp_dir}")
                # ---- MODIFICATION END ----

                CLIENT_SUMMARY_CSV = os.path.join(exp_dir, "_client_summary.csv")
                _client_summary_fields = [
                    "experiment", "source", "model_name", "client_id",
                    "accuracy", "precision", "recall", "f1_score", "auc_score"
                ]
                with open(CLIENT_SUMMARY_CSV, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=_client_summary_fields).writeheader()

                data = all_numpy_data[source_name]
                class_labels = SOURCE_CLASS_LABELS[source_name]
                num_classes = len(class_labels)

                client_dls, server_val = build_fed_loaders_trad(
                    data['X_train'], data['y_train'], data['X_test'], data['y_test'],
                    batch_size=BATCH_SIZE, num_clients=n_clients,
                    stratified=stratified_flag, random_state=RANDOM_STATE
                )
                client_dls_active = _subset_clients_for_participation(client_dls, prate, seed=RANDOM_STATE)
                _print_split_diag_trad(client_dls_active, server_val, data['X_test'], class_labels)
                _save_meta(exp_dir, {
                    "clients": n_clients, "distribution": dist, "distribution_definition": DIST_DEFS.get(dist, ""),
                    "participation": prate, "method": method,
                    "client_shard_sizes": [len(dl['train'].dataset) for dl in client_dls_active],
                    "server_val_size": int(len(server_val[1])), "test_size": int(len(data['X_test'])),
                })

                model_template = TRADITIONAL_MODELS_MAP[model_name]['model']
                X_test, y_true = data['X_test'], data['y_test']

                client_models, _ = federated_training_traditional_ensemble(
                    model_template=model_template,
                    client_dataloaders=client_dls_active,
                    server_val=server_val
                )

                # --- Log client metrics ---
                for i, m in enumerate(client_models):
                    y_pred_client = m.predict(X_test)
                    probs_client = m.predict_proba(X_test) if hasattr(m, "predict_proba") else None
                    report_client = classification_report(y_true, y_pred_client, target_names=class_labels, output_dict=True, zero_division=0)
                    prec_client, rec_client, f1_client = _macro_prec_recall_from_report(report_client)
                    auc_client = _safe_auc_multiclass(y_true, probs_client, num_classes)
                    with open(CLIENT_SUMMARY_CSV, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=_client_summary_fields)
                        writer.writerow({
                            "experiment": exp_id, "source": source_name, "model_name": f"{model_name}-FedAvg",
                            "client_id": i + 1, "accuracy": (y_pred_client == y_true).mean(),
                            "precision": prec_client, "recall": rec_client, "f1_score": f1_client, "auc_score": auc_client
                        })

                # --- Evaluate server model ---
                sizes = [len(dl['train'].dataset) for dl in client_dls_active]
                shard_weights = np.array(sizes, dtype=float)
                y_pred, probs = evaluate_federated_ensemble(
                    client_models, X_test, y_test=y_true,
                    class_labels=class_labels, show_k=3, weights=shard_weights
                )
                cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
                client_cms_test = [
                    confusion_matrix(y_true, m.predict(X_test), labels=list(range(num_classes)))
                    for m in client_models
                ]

                # --- Plot Panels ---
                try:
                    server_y_true, server_y_prob = y_true, probs
                    client_pairs = _client_test_probs_trad(client_models, X_test)
                    experiment_data_dict = {
                        "title": f"{len(client_models)} clients",
                        "server_cm": cm,
                        "client_cms": client_cms_test,
                        "server_y_true": server_y_true,
                        "server_y_prob": server_y_prob,
                        "client_pairs": client_pairs,
                        "class_labels": class_labels,
                        "model_name": model_name,
                        "source_name": source_name,
                    }
                    # Column
                    plot_col_cms(experiment_data_dict, save_dir=exp_dir)
                    plot_col_rocs(experiment_data_dict, save_dir=exp_dir)
                    # Grid
                    plot_grid_cms(
                        cm, client_cms_test, exp_dir, model_name, source_name,
                        class_labels=class_labels, dpi=180
                    )
                    plot_grid_rocs(
                        server_y_true, server_y_prob, client_pairs, exp_dir, model_name, source_name,
                        class_labels=class_labels, dpi=180
                    )
                except Exception as e:
                    _warn(f"Panel plotting error: {e}")

                # --- Log server metrics ---
                report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
                macro_prec, macro_rec, macro_f1 = _macro_prec_recall_from_report(report)
                auc_macro = _safe_auc_multiclass(y_true, probs, num_classes)
                with open(SUMMARY_CSV, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=_summary_fields)
                    writer.writerow({
                        "experiment": exp_id, "source": source_name, "model_name": f"{model_name}-FedAvg",
                        "clients": n_clients, "distribution": dist, "participation": prate, "method": method,
                        "accuracy": (y_pred == y_true).mean(),
                        "precision": macro_prec, "recall": macro_rec, "f1_score": macro_f1,
                        "auc_score": auc_macro, "save_dir": exp_dir
                    })

print("\n✅ Experiment loop complete. All artifacts saved under:", EXP_BASE_DIR)
print("  Summary CSV:", SUMMARY_CSV)