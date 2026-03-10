# ===================================================================
# Standard Library Imports
# ===================================================================
import copy
import csv
import importlib
import os
import random
import sys
import time
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===================================================================
# Third-Party Library Imports
# ===================================================================

# ---- Core Scientific Libraries ----
import numpy as np # type: ignore
import pandas as pd # type: ignore

# ---- Visualization ----
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from matplotlib.transforms import ScaledTranslation # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore

# ---- Image Processing ----
from PIL import Image # type: ignore
from skimage.feature import hog # type: ignore

# ---- PyTorch / Deep Learning ----
import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.optim import AdamW, Optimizer # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from torchvision import transforms # type: ignore

# ---- Scikit-learn ----
from sklearn.base import clone # type: ignore
from sklearn.calibration import CalibratedClassifierCV # type: ignore
from sklearn.decomposition import PCA, KernelPCA # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.feature_selection import VarianceThreshold # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, auc, classification_report,
    confusion_matrix, f1_score, log_loss, roc_auc_score, roc_curve
) # type: ignore
from sklearn.model_selection import (
    GridSearchCV, ParameterGrid, StratifiedKFold, cross_validate,
    learning_curve, train_test_split
) # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.pipeline import FeatureUnion, Pipeline # type: ignore
from sklearn.preprocessing import (
    FunctionTransformer, LabelEncoder, PowerTransformer, QuantileTransformer,
    StandardScaler, label_binarize
) # type: ignore
from sklearn.svm import SVC, LinearSVC # type: ignore

# ---- Other Utilities ----
from joblib import Memory # type: ignore

# ===================================================================
# Optional / Guarded Imports
# ===================================================================
try:
    from tqdm.auto import tqdm  # Works in notebooks & terminals
except ImportError:
    from tqdm import tqdm # type: ignore

try:
    import cv2 # type: ignore
except ImportError:
    cv2 = None

try:
    import timm  # For pretrained CNN models
except ImportError:
    timm = None

try:
    # Optional: Keras for image I/O and augmentation
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator, img_to_array, load_img
    )
except ImportError:
    ImageDataGenerator = None
    img_to_array = None
    load_img = None

# ===================================================================
# Project-Specific Utilities & Settings
# ===================================================================
def set_seed(seed: int = 42) -> None:
    """Sets the seed for reproducibility across different libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
    except NameError:
        pass  # PyTorch not imported

# ---- Matplotlib / Seaborn Default Styles ----
sns.set(context="notebook", style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (8, 5)

print("✅ Imports and setup complete.")

# ===================================================================
# Configuration
# ===================================================================

## Environment and Device Setup
# -------------------------------------------------------------------
def get_device():
    """Detects and returns the best available hardware device."""
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

DEVICE = get_device()
ON_COLAB = "google.colab" in sys.modules
ON_MAC = sys.platform == "darwin"
RANDOM_STATE = 42
set_seed(RANDOM_STATE)


from pathlib import Path

# -------------------------------------------------
# Project root (one level above src/)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Project Root: {PROJECT_ROOT}")
# -------------------------------------------------
# Dataset paths (adjust if your data lives elsewhere)
# -------------------------------------------------
ORIG_ROOT = PROJECT_ROOT / "data" / "Cotton Disease"
GEOM_ROOT = PROJECT_ROOT / "data" / "augmented_cotton_dataset_v2"

# -------------------------------------------------
# Results / experiment output
# -------------------------------------------------
SAVE_DIR = PROJECT_ROOT / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Dataset registry
# -------------------------------------------------
DATASETS = {
    "Original": ORIG_ROOT,
    "Geometric": GEOM_ROOT,
}


## Run Control and Dataset Selection
# -------------------------------------------------------------------
RUN_MODE = "both" # Options: "deep", "traditional", "both"
RUN_ON_DATASET = "Geometric"  # Options: "Original", "Geometric", "Both"
RUN_FEDERATED_LEARNING = True  # Master switch for Federated Learning

## Model Selection and Definitions
# -------------------------------------------------------------------
# Use the commented-out lists for a full run
#DEEP_MODELS = ["densenet121"]
DEEP_MODELS = ["vgg16", "resnet50", "mobilenetv2_100", "efficientnet_b0", "inception_v3", "densenet121"]

#TRADITIONAL_MODELS = ["Random Forest"]
TRADITIONAL_MODELS = ["SVM", "Naïve Bayes", "KNN", "Random Forest"]

TRADITIONAL_MODELS_MAP = {
    "LogReg": {"model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)},
    "SVM": {"model": SVC(random_state=RANDOM_STATE, probability=True)},
    "Random Forest": {"model": RandomForestClassifier(random_state=RANDOM_STATE)},
    "KNN": {"model": KNeighborsClassifier()},
    "Naïve Bayes": {"model": GaussianNB()},
}

## Training and Hyperparameters
# -------------------------------------------------------------------
NUM_EPOCHS = 50
BATCH_SIZE = 64
BASE_LEARNING_RATE = 1e-3
FREEZE_BACKBONE = True
# Example: unfreeze last block at epoch 10, last two at epoch 20
UNFREEZE_SCHEDULE = {10: 1, 20: 2}

## Federated Learning Settings
# -------------------------------------------------------------------
FED_ROUNDS = 10 # Total communication rounds
FED_CLIENTS_PER_ROUND = 1 # Clients sampled per round
FED_LOCAL_EPOCHS = 2 # Local epochs per client
FED_LEARNING_RATE = 1e-3 # Learning rate for local client updates
USE_STRATIFIED_CLIENT_SPLITS = True # Ensure clients get a mix of classes

## Feature and Pipeline Settings
# -------------------------------------------------------------------
# Traditional model features
USE_FLATTENED_PIXELS_FOR_TRAD = True
PIXEL_IMG_SIZE = 32
SVM_BACKEND = "kpca" # Options: "kpca", "pca", "none"

# Caching for sklearn pipelines to speed up repeated runs
CACHE_DIR = Path("sk_cache")
memory = Memory(location=str(CACHE_DIR), verbose=0)

# Metrics for model evaluation
SCORING = {
    "acc": "accuracy",
    "f1": "f1_macro",
    "nll": "neg_log_loss",
}
REFIT_METRIC = "f1"

## Sanity Checks and Configuration Summary
# -------------------------------------------------------------------
def _warn_if_missing(p: Path, name: str):
    if not p.exists():
        print(f"⚠️  Warning: {name} path not found -> {p}")

_warn_if_missing(ORIG_ROOT, "Original dataset")
_warn_if_missing(GEOM_ROOT, "Geometric dataset")

print(
    f"\n[CONFIG SUMMARY]\n"
    f"  - Environment: {'Colab' if ON_COLAB else 'Local/Mac' if ON_MAC else 'Other'}\n"
    f"  - Device:      {DEVICE.type.upper()}\n"
    f"  - Save Dir:    {SAVE_DIR}\n"
    f"  - Run Mode:    {RUN_MODE.title()}\n"
    f"  - Dataset:     {RUN_ON_DATASET}\n"
    f"  - Federated:   {'Yes' if RUN_FEDERATED_LEARNING else 'No'}"
)