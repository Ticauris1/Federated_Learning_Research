# ===================================================================
# Standard Library Imports
# ===================================================================
import os
import random
import sys
import copy
import importlib
import time
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.feature_selection import VarianceThreshold
import csv
from math import ceil
# ===================================================================
# Third-Party Library Imports
# ===================================================================

# ---- Core Scientific Libraries ----
import numpy as np
import pandas as pd
from joblib import Memory
from matplotlib.transforms import ScaledTranslation
# ---- Visualization ----
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- Image Processing ----
from PIL import Image
from skimage.feature import hog

# ---- PyTorch / Deep Learning ----
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---- Scikit-learn ----
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
    label_binarize,
)
from sklearn.svm import SVC, LinearSVC
import os, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
# ===================================================================
# Optional / Guarded Imports
# ===================================================================
try:
    from tqdm.auto import tqdm  # Works in notebooks & terminals
except ImportError:
    from tqdm import tqdm
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import timm  # For pretrained CNN models
except ImportError:
    timm = None
try:
    # Optional: Keras for image I/O and augmentation
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator,
        img_to_array,
        load_img,
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

## File Paths
# -------------------------------------------------------------------
if ON_COLAB:
    # Assumes dataset is uploaded to the Colab environment
    ORIG_ROOT = Path("/content/Cotton Disease")
    GEOM_ROOT = Path("/content/augmented_cotton_dataset_v2")
    SAVE_DIR = Path("/content/drive/MyDrive/dr_partha_research_results/final_run")
else:
    # Update this path to your local machine's project directory
    LOCAL_PROJECT_ROOT = Path.home() / "Desktop/research_src"
    ORIG_ROOT = LOCAL_PROJECT_ROOT / "Cotton Disease"
    GEOM_ROOT = LOCAL_PROJECT_ROOT / "augmented_cotton_dataset_v2"
    SAVE_DIR = LOCAL_PROJECT_ROOT / "results"

DATASETS = {"Original": ORIG_ROOT, "Geometric": GEOM_ROOT}
SAVE_DIR.mkdir(parents=True, exist_ok=True) # Ensure save directory exists

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
FED_CLIENTS_PER_ROUND = 5 # Clients sampled per round
FED_LOCAL_EPOCHS = 5 # Local epochs per client
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