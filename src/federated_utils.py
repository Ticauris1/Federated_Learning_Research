import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# You'll need to import your model factory function from models.py
from models import make_dl_model 

# --- Copy from your 'Federated Utilities' cell ---

def _safe_auc_multiclass(y_true, probs, n_classes):
    try:
        if probs is None: return np.nan
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if y_bin.shape[1] < 2: return np.nan
        return roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        return np.nan

def _macro_prec_recall_from_report(report_dict):
    try:
        m = report_dict.get("macro avg", {})
        return m.get("precision", np.nan), m.get("recall", np.nan), m.get("f1-score", np.nan)
    except Exception:
        return np.nan, np.nan, np.nan

# --- Copy from your 'EXPERIMENT SCAFFOLD' cell ---

def _collect_client_test_probs_from_states(
    model_name, num_classes, device, last_local_state_list, test_loader
):
    """Returns a list of (y_true, probs) for each client state; uses global if state is None."""
    import torch
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

# --- Copy from your 'MAIN EXPERIMENT LOOP' cell ---

def _client_test_probs_trad(client_models, X_test):
    pairs = []
    for m in client_models:
        if hasattr(m, "predict_proba"):
            try:
                pairs.append((None, m.predict_proba(X_test)))
            except Exception:
                print(f"Warning: {type(m).__name__}.predict_proba failed; skipping client ROC.")
                pairs.append((None, None))
        else:
            print(f"Warning: Model {type(m).__name__} lacks predict_proba; skipping client ROC.")
            pairs.append((None, None))
    return pairs