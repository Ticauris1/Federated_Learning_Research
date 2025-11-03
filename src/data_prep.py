from sklearn.preprocessing import LabelEncoder # type: ignore
from torchvision import transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from config import ( DATASETS, PIXEL_IMG_SIZE, RUN_ON_DATASET,
                    DEVICE,  RUN_ON_DATASET,
                    BATCH_SIZE, RANDOM_STATE)

from data_utils import (
    ImageDataset,
    find_image_files,
    tensors_to_flat_numpy
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

