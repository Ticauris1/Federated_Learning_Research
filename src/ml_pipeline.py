import numpy as np # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn.model_selection import learning_curve # type: ignore
import pandas as pd # type: ignore
from sklearn.feature_selection import VarianceThreshold # type: ignore
from sklearn.calibration import CalibratedClassifierCV # type: ignore
from sklearn.svm import LinearSVC # type: ignore
from sklearn.decomposition import KernelPCA # type: ignore
from sklearn.preprocessing import PowerTransformer, QuantileTransformer # type: ignore
from sklearn.model_selection import ParameterGrid, cross_validate # type: ignore
from sklearn.base import clone # type: ignore
from sklearn.metrics import log_loss, accuracy_score, f1_score # type: ignore
from data_utils import tensors_to_flat_numpy # Import the one you kept
from config import RANDOM_STATE  

# ---- Safe defaults ----
SKLEARN_PIPELINES = {}
SKLEARN_PARAM_GRIDS = {}

try: RANDOM_STATE
except NameError: RANDOM_STATE = 42

try: KNN_WIDE_GRID
except NameError: KNN_WIDE_GRID = True

try: PRUNE_KNN_BY_CV
except NameError: PRUNE_KNN_BY_CV = False

try: REFIT_METRIC
except NameError: REFIT_METRIC = "f1"

try: SCORING
except NameError: SCORING = {"acc": "accuracy", "f1": "f1_macro", "nll": "neg_log_loss"}

try: GRID_N_JOBS
except NameError: GRID_N_JOBS = -1

# Backend for SVM ("svc" | "linearsvc_calibrated")
try: SVM_BACKEND
except NameError: SVM_BACKEND = "svc"   # default to RBF SVC for best accuracy

# joblib cache
try:
    memory
except NameError:
    try:
        from joblib import Memory
        memory = Memory(location="sk_cache", verbose=0)
    except Exception:
        memory = None

# Optional FEATS (e.g., HOG/Color) — leave None for pixels-only
except NameError: FEATS = None

def _maybe_feats_step(FEATS):
    if FEATS is None: return []
    if isinstance(FEATS, tuple) and len(FEATS) == 2: return [FEATS]
    return [("feats", FEATS)]

# Use a bigger flat-pixel size elsewhere if you like (training cell):
PIXEL_IMG_SIZE = globals().get("PIXEL_IMG_SIZE", 64)

class SafePCA(PCA):
    def fit(self, X, y=None):
        if isinstance(self.n_components, (int, np.integer)):
            kmax = int(min(X.shape[0], X.shape[1]))
            if self.n_components > kmax:
                self.n_components = kmax  # clip to valid range for this fold
        return super().fit(X, y)

# ================= Pipelines =================
# RBF SVC on flat pixels (strong baseline)
SVMPixels_svc = Pipeline([
    ("vt", VarianceThreshold(1e-4)),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(n_components=128, whiten=True, svd_solver="randomized", random_state=RANDOM_STATE)),
    ("clf", SVC(
        kernel="rbf",
        probability=True,                # leave True so 'neg_log_loss' works in CV
        class_weight="balanced",
        cache_size=1024,
        tol=1e-3,
        shrinking=True,
        random_state=RANDOM_STATE
    )),
], memory=memory)

# Optional FEATS+SVC variant (rarely needed on your data)
SVM_svc_with_feats = Pipeline([
    *_maybe_feats_step(FEATS),
    ("vt", VarianceThreshold(1e-4)),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(n_components=128, svd_solver="randomized", random_state=RANDOM_STATE)),
    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
], memory=memory)

# LinearSVC + calibration (fast linear baseline)
SVMPixels_linCal = Pipeline([
    ("vt", VarianceThreshold(1e-4)),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(n_components=64, svd_solver="randomized", random_state=RANDOM_STATE)),
    ("clf", CalibratedClassifierCV(
        estimator=LinearSVC(
            dual=False,
            class_weight="balanced",
            tol=1e-3,
            max_iter=8000,
            random_state=RANDOM_STATE
        ),
        method="sigmoid",
        cv=3
    )),
], memory=memory)

SVMPixels_kpca = Pipeline([
    ("vt", VarianceThreshold(1e-4)),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("kpca", KernelPCA(kernel="rbf", random_state=RANDOM_STATE)),
    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
], memory=memory)

# KNN / RF / NB (unchanged bests)
SKLEARN_PIPELINES.update({
    "KNN": Pipeline([
        ("vt", VarianceThreshold(1e-4)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=384, svd_solver="randomized", random_state=RANDOM_STATE)),
        ("clf", KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="cosine", algorithm="brute", leaf_size=30, n_jobs=-1
        )),
    ], memory=memory),
    "Random Forest": Pipeline([
        *_maybe_feats_step(FEATS),
        ("vt", VarianceThreshold(1e-4)),
        ("pca", PCA(n_components=128, svd_solver="randomized", random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", random_state=RANDOM_STATE)),
    ], memory=memory),
    "Naïve Bayes": Pipeline([
        *_maybe_feats_step(FEATS),
        ("vt", VarianceThreshold(1e-4)),
        ("gaussify", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("pca", PCA(n_components=64, whiten=True, svd_solver="randomized", random_state=RANDOM_STATE)),
        ("clf", GaussianNB()),
    ], memory=memory),
})


# Safety: FEATS injection appears at most once
for k, pipe in SKLEARN_PIPELINES.items():
    feats_steps = [s for s in pipe.steps if s[0] == "feats"]
    assert len(feats_steps) <= 1, f"{k}: FEATS appears more than once"
    if feats_steps:
        assert not isinstance(feats_steps[0][1], tuple), f"{k}: FEATS is double-wrapped"
print("✅ FEATS integration looks good.")

# ============== Parameter grids ==============
# Focused RBF SVC grid (around your winners)

SVM_SVC_PARAM_GRID_FOCUSED = [
    {
        "pca__n_components": [32, 0.90, 0.95, 0.99],  # #"pca__n_components": [128, 192, 256],
        "clf__C": [1, 3, 10, 30],
        "clf__gamma": ["scale", 3e-3, 1e-2],
    }
]

# LinearSVC + calibration grid
SVM_LINSVC_PARAM_GRID = [
    {"pca__n_components": [32, 0.90, 0.95, 0.99], "clf__estimator__C": [0.25, 1.0, 4.0, 16.0]}, #{"pca__n_components": [32, 64, 96, 128],
]

SVM_KPCA_PARAM_GRID = [
    {
        "kpca__n_components": [32, 64, 90],
        "kpca__gamma": [0.01, 0.1, None], # None uses 1/n_features
        "clf__C": [1, 10, 100],
    }
]

KNN_PARAM_GRID_MICRO = [{
    "pca__n_components": [32, 0.90, 0.95, 0.99],#"pca__n_components": [256, 384],
    "clf__n_neighbors": [3, 5, 7, 11],
    "clf__weights": ["distance"],
    "clf__metric": ["cosine"],
    "clf__algorithm": ["brute"],
    "clf__leaf_size": [15, 30, 60],
}]
KNN_PARAM_GRID_WIDE = [
    {"pca__n_components": [32, 0.90, 0.95, 0.99],#"pca__n_components": [128, 256, 384], "clf__metric": ["minkowski"],
     "clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"],
     "clf__algorithm": ["auto"]},
    {"pca__n_components": [32, 0.90, 0.95, 0.99],#"pca__n_components": [128, 256, 384], "clf__metric": ["chebyshev", "cosine"],
     "clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"],
     "clf__algorithm": ["auto", "brute"]},
]
RF_PARAM_GRID = {
    "pca__n_components": [32, 0.90, 0.95, 0.99],  # "pca__n_components": [128, 256],
    "clf__n_estimators": [300, 600],
    "clf__max_depth": [None, 40],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt", 0.5],
}
NB_PARAM_GRID = {
    "gaussify": [PowerTransformer(), QuantileTransformer(output_distribution="normal", subsample=2_00_000, random_state=RANDOM_STATE)],
    "pca__whiten": [True, False],               # whitening often helps NB
    "pca__n_components": [32, 0.90, 0.95, 0.99],
    "clf__var_smoothing": np.logspace(-10, -1, 12),  # wider
}
# Register non-SVM grids
SKLEARN_PARAM_GRIDS.update({
    "KNN": (KNN_PARAM_GRID_WIDE if KNN_WIDE_GRID else KNN_PARAM_GRID_MICRO),
    "Random Forest": RF_PARAM_GRID,
    "Naïve Bayes": NB_PARAM_GRID,
})

# Choose SVM backend
if SVM_BACKEND == "svc":
    SKLEARN_PIPELINES["SVM"] = SVMPixels_svc
    SKLEARN_PARAM_GRIDS["SVM"] = SVM_SVC_PARAM_GRID_FOCUSED
    print("✅ SVM backend: SVC (RBF) with PCA | Pixels-only")
elif SVM_BACKEND == "linearsvc_calibrated":
    SKLEARN_PIPELINES["SVM"] = SVMPixels_linCal
    SKLEARN_PARAM_GRIDS["SVM"] = SVM_LINSVC_PARAM_GRID
    print("✅ SVM backend: LinearSVC + Calibration with PCA | Pixels-only")
# --- NEW: Added 'kpca' option to the selection logic ---
elif SVM_BACKEND == "kpca":
    SKLEARN_PIPELINES["SVM"] = SVMPixels_kpca
    SKLEARN_PARAM_GRIDS["SVM"] = SVM_KPCA_PARAM_GRID
    print("✅ SVM backend: SVC (RBF) with KernelPCA | Pixels-only")
else:
    raise ValueError("SVM_BACKEND must be 'svc', 'linearsvc_calibrated', or 'kpca'.")
# ============== Learning curve history generator ==============
def generate_learning_curve_history(estimator, X, y, cv=3):
    """
    Generates a learning curve for a scikit-learn model and formats it
    as a history dictionary, similar to Keras/PyTorch.

    Returns a list of dictionaries [{'epoch':, 'train_acc':, 'val_acc':}].
    """
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,  # Use all available CPUs
        train_sizes=np.linspace(0.1, 1.0, 10) # Test 10 different training set sizes
    )

    # Calculate mean scores
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)

    # Format into the desired history structure
    history_records = []
    for i, train_size in enumerate(train_sizes):
        history_records.append({
            'epoch': train_size, # Using train_size as the 'epoch'
            'train_acc': train_scores_mean[i],
            'val_acc': validation_scores_mean[i]
        })

    return history_records

# ============== Trainer with old-school (GridSearch-like) prints ==============
USE_OLD_STYLE_PRINTS = True

def _inline_progress(prefix, i, total, best_val):
    best_show = best_val if np.isfinite(best_val) else float("-inf")
    print(f"\r{prefix} candidates {total} | trying {i}/{total} | best {best_show:.4f}",
          end="", flush=True)

def _pp_params_one_line(d):
    if not d: return "{}"
    items = sorted(d.items(), key=lambda kv: kv[0])
    return "{" + ", ".join([f"{k}={v}" for k, v in items]) + "}"

def _normalize_refit(refit):
    m = {"accuracy":"acc","acc":"acc","f1":"f1","f1_macro":"f1","neg_log_loss":"nll","nll":"nll"}
    return m.get(refit, "f1")

def train_sklearn_with_pipeline_on_pixels(
    model_key,
    X_train, y_train,
    X_test, y_test,
    cv=3,
    scoring=SCORING,
    refit=REFIT_METRIC,
    n_jobs=GRID_N_JOBS,
    error_score="raise",
):
    pipe = SKLEARN_PIPELINES[model_key]
    grid = SKLEARN_PARAM_GRIDS.get(model_key, [])

    # Optional KNN pruning hook
    try:
        if model_key == "KNN" and PRUNE_KNN_BY_CV:
            grid = _prune_knn_grid_for_cv(grid, n_train=len(X_train), cv=cv)
    except NameError:
        pass

    # Flatten grids
    if isinstance(grid, dict):
        param_list = list(ParameterGrid(grid))
    else:
        param_list = []
        for g in grid:
            param_list.extend(list(ParameterGrid(g)))

    refit = _normalize_refit(refit)
    print(f"\n[{model_key}] Starting search (cv={cv}, refit='{refit}', n_jobs={n_jobs})")

    # No grid → fit once
    if len(param_list) == 0:
        best = clone(pipe).fit(X_train, y_train)
        y_pred = best.predict(X_test)
        try:
            proba = best.predict_proba(X_test)
            test_nll = log_loss(y_test, proba, labels=np.unique(y_train))
        except Exception:
            proba, test_nll = None, None
        test_acc = accuracy_score(y_test, y_pred)
        test_f1  = f1_score(y_test, y_pred, average="macro")
        print(f"[{model_key}] Best params: {_pp_params_one_line(None)}")
        print(f"[{model_key}] Test Accuracy: {test_acc:.4f} | Test F1(macro): {test_f1:.4f}"
              + (f" | Test NLL: {test_nll:.4f}" if test_nll is not None else ""))
        return best, proba, y_test, pd.DataFrame([]), {"acc": test_acc, "f1": test_f1, "nll": test_nll}

    rows, best_score, best_params = [], -np.inf, None

    for i, params in enumerate(param_list, start=1):
        est = clone(pipe).set_params(**params)
        try:
            cv_res = cross_validate(
                est, X_train, y_train,
                scoring=scoring, cv=cv, n_jobs=n_jobs,
                return_train_score=False, error_score=error_score
            )
        except Exception:
            _inline_progress(f"[{model_key}]", i, len(param_list), best_score); continue

        te_acc = np.mean(cv_res.get("test_acc", np.array([np.nan])))
        te_f1  = np.mean(cv_res.get("test_f1",  np.array([np.nan])))
        te_nll = -np.mean(cv_res.get("test_nll", np.array([np.nan])))  # flip sign back to +NLL

        rows.append({"params": params, "mean_test_acc": te_acc, "mean_test_f1": te_f1, "mean_test_nll": te_nll})

        refit_val = {"acc": te_acc, "f1": te_f1, "nll": (-te_nll if np.isfinite(te_nll) else -np.inf)}[refit]
        if np.isfinite(refit_val) and refit_val > best_score:
            best_score, best_params = refit_val, params

        _inline_progress(f"[{model_key}]", i, len(param_list), best_score)

    print()  # newline

    cv_df = pd.DataFrame(rows)
    for col in ["mean_test_acc", "mean_test_f1", "mean_test_nll"]:
        if col not in cv_df.columns: cv_df[col] = np.nan

    sort_key = {"acc":"mean_test_acc","f1":"mean_test_f1","nll":"mean_test_nll"}[refit]
    if sort_key not in cv_df.columns: sort_key = "mean_test_acc"
    cv_df = cv_df.sort_values(by=sort_key, ascending=(refit=="nll"), na_position="last")

    print(f"[{model_key}] Best params: {_pp_params_one_line(best_params)}")

    best = clone(pipe if best_params is None else pipe.set_params(**best_params)).fit(X_train, y_train)

    y_pred  = best.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average="macro")
    try:
        proba = best.predict_proba(X_test)
        test_nll = log_loss(y_test, proba, labels=np.unique(y_train))
    except Exception:
        proba, test_nll = None, None

    print(f"[{model_key}] Test Accuracy: {test_acc:.4f} | Test F1(macro): {test_f1:.4f}"
          + (f" | Test NLL: {test_nll:.4f}" if test_nll is not None else ""))

    return best, proba, y_test, cv_df, {"acc": test_acc, "f1": test_f1, "nll": test_nll}
