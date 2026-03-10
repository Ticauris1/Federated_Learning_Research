import os
import string
from config import DEEP_MODELS, TRADITIONAL_MODELS
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from itertools import cycle
from sklearn.preprocessing import label_binarize # type: ignore
from sklearn.metrics import roc_curve, auc  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable # type: ignore
from matplotlib.transforms import ScaledTranslation # type: ignore
# Also add the missing functions: _warn, _assert_and_log_save, _verify_saved
def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax."""
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.clip(expZ.sum(axis=1, keepdims=True), 1e-12, None)

def _warn(msg):  # uniform warn printer
    print(f"[warn] {msg}")

def _assert_and_log_save(fig, path):
    """Saves a figure and closes it, creating the directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")

def _verify_saved(paths):
    """Checks if a list of file paths exists, warns if not."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        _warn("Missing expected outputs:\n  - " + "\n  - ".join(missing))
    else:
        print("[ok] all expected files saved.")

# --- plotting drawers (ROC/CM minimal) + wrappers ---
def draw_roc_on_ax(ax, y_true, y_prob, title, letter="a", letter_pos="bottom",
                   class_labels=None, legend_fontsize=8, legend_loc="lower right", grid=True):
    if letter_pos == 'top': plot_title, x_label = f"({letter}) {title}", 'False Positive Rate'
    else:                   plot_title, x_label = title, f'False Positive Rate\n({letter})'
    if (y_true is None) or (y_prob is None):
        ax.text(0.5,0.5,"No data",ha="center",va="center",style='italic',color='gray')
        ax.set_title(plot_title); ax.set_xticks([]); ax.set_yticks([]); return
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = y_true_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    palette = cycle(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f'])
    for i, color in zip(range(n_classes), palette):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        lab = class_labels[i] if class_labels is not None else f"Class {i}"
        ax.plot(fpr[i], tpr[i], color=color, lw=1.5, label=f'{lab} (AUC={roc_auc[i]:.2f})')
    ax.plot([0,1],[0,1],color='navy',lw=1.2,linestyle='--')
    ax.set_xticks(np.arange(0,1.1,0.2)); ax.set_yticks(np.arange(0,1.1,0.2))
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    if grid: ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_ylabel('True Positive Rate'); ax.set_xlabel(x_label); ax.set_title(plot_title)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=False)

def draw_cm_on_ax(ax, cm, title, letter="b", letter_pos="bottom",
                  class_labels=None, cmap=plt.cm.Blues, annotate=True):
    if cm is None or (hasattr(cm,"size") and cm.size==0):
        ax.text(0.5,0.5,"No data",ha="center",va="center",style='italic',color='gray')
        if letter_pos!='top': ax.set_xlabel(f'({letter})')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title); return
    acc = (np.trace(cm) / max(cm.sum(), 1)) if cm.ndim==2 else 0.0
    plot_title = f"{title}\nAccuracy: {acc:.2f}" if letter_pos!='top' else f"({letter}) {title}\nAccuracy: {acc:.2f}"
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap); ax.set_title(plot_title)
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="4%", pad=0.1)
    ax.get_figure().colorbar(im, cax=cax, label='Count')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel('True'); ax.set_xlabel('Predicted' if letter_pos=='top' else f'Predicted\n({letter})')
    if annotate:
        thresh = cm.max() / 2.0 if cm.max()>0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j,i,f"{cm[i,j]:d}",ha="center",va="center",
                        color=("white" if cm[i,j]>thresh else "black"), fontsize=9)

'''
# --- Confusion-matrix drawer (compat wrapper) ---
def _draw_cm_compat(ax, cm, title, *, letter="b", letter_pos="bottom", class_labels=None):
    """
    Calls your existing draw_cm_on_ax with class_labels if supported;
    otherwise falls back to a call without that kwarg.
    """
    try:
        # Preferred (newer) signature
        return draw_cm_on_ax(ax, cm, title, letter=letter, letter_pos=letter_pos, class_labels=class_labels)
    except TypeError:
        # Older signature (no class_labels kwarg)
        return draw_cm_on_ax(ax, cm, title, letter=letter, letter_pos=letter_pos)
'''
def plot_fed_server_roc_cm(model_name, source_name, y_true, y_prob, cm, class_labels, out_dir,
                           roc_letter="a", cm_letter="b", dpi=200):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2,4.5), dpi=dpi, constrained_layout=True)
    draw_roc_on_ax(ax, y_true, y_prob, f"{model_name} — {source_name} (Server ROC)", letter=roc_letter, class_labels=class_labels)
    fig.savefig(os.path.join(out_dir, f"{source_name}_{model_name}_FedAvg_server_ROC.png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.8,4.4), dpi=dpi, constrained_layout=True)
    draw_cm_on_ax(ax, cm, f"{model_name} — {source_name} (Server CM)", letter=cm_letter, class_labels=class_labels)
    fig.savefig(os.path.join(out_dir, f"{source_name}_{model_name}_FedAvg_server_CM.png")); plt.close(fig)

def plot_federated_client_cms(client_cms, save_dir, model_name, source_name, class_labels):
    n = len(client_cms)
    if not n:
        print("[Client CM Panel] none"); return
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5.5, rows*5), constrained_layout=True, dpi=150)
    axes = axes.flatten(); fig.suptitle(f'Final Confusion Matrices for Clients ({model_name} on {source_name})', fontsize=16, weight='bold')
    for i in range(rows*cols):
        ax = axes[i]
        if i < n: _draw_cm_compat(ax, client_cms[i], f"Client {i+1}", class_labels=[str(j) for j in range(len(class_labels))])
        else: ax.axis('off')
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{source_name}_{model_name}_FedAvg_client_cms.png"), bbox_inches="tight", dpi=200); plt.close(fig)

def plot_federated_learning_curves(global_history, client_histories, save_dir, model_name, source_name):
    rounds = range(1, len(global_history['accuracy']) + 1)
    avg_client_acc, avg_client_loss = [], []
    num_clients = len(client_histories)
    for r in range(len(rounds)):
        accs = [client_histories[i]['accuracy'][r] for i in range(num_clients) if len(client_histories[i]['accuracy']) > r]
        losses = [client_histories[i]['loss'][r] for i in range(num_clients) if len(client_histories[i]['loss']) > r]
        avg_client_acc.append(np.mean(accs) if accs else np.nan)
        avg_client_loss.append(np.mean(losses) if losses else np.nan)
    for metric, client_data, global_data in [('Accuracy', avg_client_acc, global_history['accuracy']),
                                             ('Loss', avg_client_loss, global_history['loss'])]:
        plt.figure(figsize=(10,6))
        plt.plot(rounds, global_data, '-', label='Global (Server)')
        plt.plot(rounds, client_data, '--', label='Avg Client')
        plt.title(f'Federated {metric}: {model_name} on {source_name}')
        plt.xlabel('Communication Round'); plt.ylabel(metric)
        plt.legend(); plt.grid(True); plt.xticks(rounds); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{source_name}_{model_name}_FedAvg_{metric.lower()}_curve.png")); plt.close()

def plot_federated_per_client_curves(client_histories, save_dir, model_name, source_name):
    num_clients = len(client_histories)
    if not num_clients: return
    rows, cols = 3, 3
    total_rounds = 0
    for i in range(num_clients): total_rounds = max(total_rounds, len(client_histories.get(i, {}).get('accuracy', [])))
    rounds = range(1, total_rounds + 1)

    # Accuracy panel
    fig_acc, axes_acc = plt.subplots(rows, cols, figsize=(12,8), constrained_layout=True, dpi=150)
    axes_acc = axes_acc.flatten(); fig_acc.suptitle(f'Per-Client Federated Accuracy: {model_name} on {source_name}', fontsize=16, weight='bold')
    for i in range(rows*cols):
        ax = axes_acc[i]
        if i < num_clients:
            acc_hist = client_histories.get(i, {}).get('accuracy', [])
            plot_data = np.full(total_rounds, np.nan); plot_data[:len(acc_hist)] = acc_hist
            ax.plot(rounds, plot_data, '-'); ax.set_title(f'Client {i+1}'); ax.grid(True); ax.set_ylim(0, 1.0)
        else: ax.axis('off')
        ax.set_xlabel('Communication Round'); ax.set_ylabel('Accuracy')
    plt.savefig(os.path.join(save_dir, f"{source_name}_{model_name}_FedAvg_panel_client_accuracy.png")); plt.close(fig_acc)

    # Loss panel
    fig_loss, axes_loss = plt.subplots(rows, cols, figsize=(12,8), constrained_layout=True, dpi=150)
    axes_loss = axes_loss.flatten(); fig_loss.suptitle(f'Per-Client Federated Loss: {model_name} on {source_name}', fontsize=16, weight='bold')
    for i in range(rows*cols):
        ax = axes_loss[i]
        if i < num_clients:
            loss_hist = client_histories.get(i, {}).get('loss', [])
            plot_data = np.full(total_rounds, np.nan); plot_data[:len(loss_hist)] = loss_hist
            ax.plot(rounds, plot_data, '-'); ax.set_title(f'Client {i+1}'); ax.grid(True)
        else: ax.axis('off')
        ax.set_xlabel('Communication Round'); ax.set_ylabel('Loss')
    plt.savefig(os.path.join(save_dir, f"{source_name}_{model_name}_FedAvg_panel_client_loss.png")); plt.close(fig_loss)

# ===================================================================
# TITLE & NAMING HELPERS
# ===================================================================
def _bar_title(n_clients: int) -> str:
    """Creates the 'Server | Client 1 | ...' string."""
    parts = ["Server"] + [f"Client {i}" for i in range(1, n_clients + 1)]
    return " | ".join(parts)

def _resource_suffix(resource_label: str | None) -> str:
    """Creates ' (Resource Label)' if provided."""
    return f" ({resource_label})" if resource_label else ""

def _infer_resource_label(model_name: str, resource_label: str | None = None) -> str | None:
    """
    Decide 'High Resources' for deep learning models and 'Low Resources'
    for traditional models, unless an explicit resource_label is provided.
    Falls back to None if model_name doesn't match known lists.
    """
    if resource_label:  # explicit wins
        return resource_label
    try:
        # Assumes DEEP_MODELS and TRADITIONAL_MODELS are globally defined
        if 'DEEP_MODELS' in globals() and model_name in DEEP_MODELS:
            return "High Resources"
        if 'TRADITIONAL_MODELS' in globals() and model_name in TRADITIONAL_MODELS:
            return "Low Resources"
    except Exception:
        pass
    return None

def _compose_suptitle(
    title: str,
    *,
    model_name: str,
    source_name: str,
    n_clients: int,
    resource_label: str | None = None,
    title_override: str | None = None,
):
    """
    Minimal, keyword-only suptitle used by all panels.

    If title_override is provided, it's returned verbatim.
    Otherwise, we return a standard "Server | Client 1 | ... [ (Resource) ]" bar.
    """
    if title_override:
        return title_override
    # append inferred/explicit resource suffix, but otherwise keep the clean bar
    suffix = _resource_suffix(_infer_resource_label(model_name, resource_label))
    return f"{_bar_title(n_clients)}{suffix}"

# ===================================================================
# CORE PLOTTING DRAWERS & STYLERS
# ===================================================================
def _place_letter_bottom(ax, letter, dy_axes=-0.16, pad_pts=34, fontsize=12):
    """
    Place a subplot letter BELOW the axes with extra padding from the x-axis.
    """
    trans = ax.transAxes + ScaledTranslation(0, -pad_pts/72.0, ax.figure.dpi_scale_trans)
    ax.text(0.5, dy_axes, f"({letter})", transform=trans,
            ha="center", va="top", fontsize=fontsize)

def _style_confusion_ax(
    ax, *, title=None, title_fs=16,
    tick_fs=None, tick_fs_x=12, tick_fs_y=12,   # per-axis tick sizes
    anno_fs=15, cbar_fs=14
):
    """(DEPRECATED by _draw_cm_compat) Style a confusion-matrix axis."""
    ax.set_title(title if title is not None else ax.get_title(), fontsize=title_fs)

    if tick_fs is not None:
        ax.tick_params(axis='both', which='both', labelsize=tick_fs)
    else:
        ax.tick_params(axis='x', which='both', labelsize=tick_fs_x)
        ax.tick_params(axis='y', which='both', labelsize=tick_fs_y)

    if ax.images:
        mappable = ax.images[0]
        try:
            for cax in ax.figure.axes:
                cb = getattr(cax, "colorbar", None)
                if cb is not None and cb.mappable is mappable:
                    cax.tick_params(labelsize=cbar_fs)
        except Exception:
            pass
    for t in ax.texts:
        try:
            t.set_fontsize(anno_fs)
        except Exception:
            pass

def _bump_cm_cell_font(ax, anno_fs=20, color=None, weight=None):
    """
    Increase the font size (and optionally color/weight) of the per-cell
    annotations drawn inside a confusion matrix heatmap on `ax`.
    """
    for t in ax.texts:
        try:
            t.set_fontsize(anno_fs)
            if color is not None:
                t.set_color(color)
            if weight is not None:
                t.set_fontweight(weight)
        except Exception:
            pass

def _thicken_roc_lines(ax, lw=3.5, tick_fs=14, label_fs=15, legend_fs=14, title_fs=18, roc_color=None):
    """Thicken every ROC line on an axes, optionally recolor to a single hue."""
    for ln in ax.lines:
        try:
            ln.set_linewidth(lw)
            if roc_color is not None:
                ln.set_color(roc_color)
        except Exception:
            pass
    ax.tick_params(axis='both', which='both', labelsize=tick_fs)
    if ax.get_xlabel(): ax.set_xlabel(ax.get_xlabel(), fontsize=label_fs)
    if ax.get_ylabel(): ax.set_ylabel(ax.get_ylabel(), fontsize=label_fs)
    if ax.get_title():  ax.set_title(ax.get_title(), fontsize=title_fs)
    leg = ax.get_legend()
    if leg:
        if leg.get_title() is not None:
            leg.set_title(leg.get_title().get_text(), prop={'size': legend_fs})
        for txt in leg.get_texts():
            txt.set_fontsize(legend_fs)

def _draw_cm_compat(ax, cm, title, class_labels=None, cmap="Blues", value_fmt="d",
                    text_light="black", text_dark="white", luminance_thresh=0.5,
                    *,  # styling knobs
                    title_fs=18, label_fs=16, tick_fs_x=13, tick_fs_y=13, cbar_fs=14):
    """A robust function to draw a single, publication-quality confusion matrix."""
    cm = np.asarray(cm)
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title if title else "Confusion Matrix", fontsize=title_fs)

    cb = ax.figure.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=cbar_fs)

    n_classes = cm.shape[0]
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    if class_labels is not None and len(class_labels) == n_classes:
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted label", fontsize=label_fs)
    ax.set_ylabel("True label", fontsize=label_fs)
    ax.tick_params(axis='x', which='both', labelsize=tick_fs_x, length=0)
    ax.tick_params(axis='y', which='both', labelsize=tick_fs_y, length=0)

    norm = im.norm
    _cmap = im.cmap
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            r, g, b, _ = _cmap(norm(val))
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            txt_color = text_dark if lum < luminance_thresh else text_light
            ax.text(j, i, f"{val:{value_fmt}}", ha="center", va="center", color=txt_color)

def _draw_multiclass_roc(ax, y_true, y_prob, title, class_labels=None):
    """Draw per-class ROC curves plus a BLACK dashed diagonal baseline."""
    if y_prob is None:
        _warn("No probabilities available; skipping ROC curve.")
        ax.set_title(title if title else "ROC (no probabilities)")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        return

    y_true = np.asarray(y_true).astype(int)
    n_classes = y_prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        name = class_labels[c] if (class_labels and c < len(class_labels)) else f"Class {c}"
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="black") # BLACK dashed baseline
    ax.set_title(title if title else "ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=True)

# ===================================================================
# COMPOSITE PLOT: CURVE OVERLAYS
# ===================================================================
def plot_server_with_clients_one_axes(
    *, global_history, client_histories, save_dir, model_name, source_name, metric="loss",
    lw=2.8, title_fs=20, tick_fs=14, label_fs=15, legend_fs=14
):
    """Overlay server + clients for a single metric."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=180)
    xg = np.arange(1, len(global_history[metric]) + 1)
    ax.plot(xg, global_history[metric], label=f"Server {metric.title()}", linewidth=lw)

    for cid, h in client_histories.items():
        if metric in h:
            # Pad history if it's shorter (e.g., client not selected every round)
            h_metric = h[metric]
            if len(h_metric) < len(xg):
                h_metric = np.pad(h_metric, (0, len(xg) - len(h_metric)), 'edge')
            ax.plot(xg, h_metric, alpha=0.85, linewidth=lw-0.6, label=f"Client {cid+1} {metric}") # Use cid+1 for 1-indexing

    ax.set_xlabel("Round", fontsize=label_fs)
    ax.set_ylabel(metric.title(), fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)

    n_clients = len(client_histories)
    overlay_title = _compose_suptitle(
        title="Overlay",
        model_name=model_name,
        source_name=source_name,
        n_clients=n_clients
    )
    ax.set_title(overlay_title, fontsize=title_fs)

    leg = ax.legend(frameon=True)
    if leg and leg.get_title():
        leg.set_title(leg.get_title().get_text(), prop={'size': legend_fs})
    for t in leg.get_texts():
        t.set_fontsize(legend_fs)

    out = os.path.join(save_dir, f"{source_name}_{model_name}_server_clients_{metric}_overlay.png")
    _assert_and_log_save(fig, out)
    print(f"[overlay] {metric.title()} -> {out}")
    return out

# ===================================================================
# COMPOSITE PLOTS: COLUMN LAYOUTS (Vertically Stacked)
# ===================================================================
def plot_col_cms(
    experiment, save_dir, dpi=280, filename=None,
    *, cell_fs=12, letter_fs=13, letter_pad_pts=34,
    suptitle_fs=18, resource_label=None, cm_cmap="Blues"
):
    """Plots [Server, Client 1, Client 2, ...] as a single vertical column."""
    import os
    import string
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    server_cm = np.asarray(experiment["server_cm"], dtype=int)
    client_cms_raw = list(experiment.get("client_cms", []))
    class_labels = experiment.get("class_labels")
    model_name = experiment.get("model_name", "Model")
    source_name = experiment.get("source_name", "Dataset")

    n_classes = len(class_labels) if class_labels is not None else None

    client_cms = []
    for idx, entry in enumerate(client_cms_raw):
        # Case 1: already a confusion matrix
        if isinstance(entry, np.ndarray) and entry.ndim == 2:
            cm = np.asarray(entry, dtype=int)

        # Case 2: tuple/list like (y_true, y_prob) or (y_true, y_pred)
        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            y_true, y_second = entry
            y_true = np.asarray(y_true)
            y_second = np.asarray(y_second)

            if y_second.ndim == 2:
                y_pred = np.argmax(y_second, axis=1)
            elif y_second.ndim == 1:
                y_pred = y_second
            else:
                raise ValueError(
                    f"Unsupported client entry format at index {idx}: "
                    f"second item shape={y_second.shape}"
                )

            labels = np.arange(n_classes) if n_classes is not None else np.unique(np.concatenate([y_true, y_pred]))
            cm = confusion_matrix(y_true, y_pred, labels=labels).astype(int)

        else:
            arr = np.asarray(entry)
            raise ValueError(
                f"Client entry at index {idx} is not a valid confusion matrix or "
                f"(y_true, y_prob)/(y_true, y_pred) pair. Got shape={arr.shape}"
            )

        if cm.ndim != 2:
            raise ValueError(f"Client {idx} confusion matrix is not 2D. Got shape={cm.shape}")

        client_cms.append(cm)

    n_clients = len(client_cms)
    nplots = 1 + n_clients

    fig, axes = plt.subplots(
        nrows=nplots, ncols=1,
        figsize=(6.0, 4.6 * nplots),
        dpi=dpi, squeeze=False
    )
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)
    cfs = int(round(cell_fs))

    # Server
    _draw_cm_compat(
        axes[0], server_cm, "Server (Global)",
        class_labels=class_labels, cmap=cm_cmap,
        title_fs=18, label_fs=16, tick_fs_x=13, tick_fs_y=13, cbar_fs=14
    )
    _bump_cm_cell_font(axes[0], anno_fs=cfs, color=None)
    _place_letter_bottom(axes[0], letters[0], fontsize=letter_fs, pad_pts=letter_pad_pts)

    # Clients
    for i, cm in enumerate(client_cms, start=0):
        ax_idx = i + 1
        _draw_cm_compat(
            axes[ax_idx], cm, f"Client {i+1}",
            class_labels=class_labels, cmap=cm_cmap,
            title_fs=18, label_fs=16, tick_fs_x=13, tick_fs_y=13, cbar_fs=14
        )
        _bump_cm_cell_font(axes[ax_idx], anno_fs=cfs, color=None)
        _place_letter_bottom(
            axes[ax_idx],
            letters[ax_idx if ax_idx < len(letters) else -1],
            fontsize=letter_fs, pad_pts=letter_pad_pts
        )

    fig.tight_layout(rect=[0, 0.18, 1, 0.97])
    fig.canvas.draw()
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    center_x = (left + right) / 2.0

    suptitle = _compose_suptitle(
        title="Confusion Matrices",
        model_name=model_name,
        source_name=source_name,
        n_clients=n_clients,
        resource_label=resource_label
    )
    fig.suptitle(suptitle, fontsize=suptitle_fs, y=0.995, x=center_x, ha="center")

    out_path = os.path.join(
        save_dir,
        filename or f"{source_name}_{model_name}_panel_confusion_matrices_col.png"
    )
    _assert_and_log_save(fig, out_path)
    print(f"[panel] CM rows×cols: {nplots}×1  -> {out_path}")
    return out_path







def plot_col_rocs(
    experiment, save_dir, dpi=180, filename=None,
    *, roc_lw=3.5, roc_color="black", tick_fs=13, label_fs=14, legend_fs=13,
    title_fs=18, letter_fs=16, letter_pad_pts=34, suptitle_fs=14,
    resource_label=None, add_black_diagonal=True
):
    """Plots [Server, Client 1, Client 2, ...] ROCs as a single vertical column."""
    server_y_true = experiment["server_y_true"]
    server_y_prob = experiment["server_y_prob"]
    client_pairs  = list(experiment.get("client_pairs", []))
    class_labels  = experiment.get("class_labels")
    model_name    = experiment.get("model_name", "Model")
    source_name   = experiment.get("source_name", "Dataset")

    n_clients = len(client_pairs)
    nplots = 1 + n_clients
    fig, axes = plt.subplots(nrows=nplots, ncols=1, figsize=(6.8, 4.8*nplots), dpi=dpi, squeeze=False)
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)

    # Server ROC
    _draw_multiclass_roc(axes[0], server_y_true, server_y_prob, "Server (Global) ROC", class_labels=class_labels)
    if add_black_diagonal:
        axes[0].plot([0, 1], [0, 1], linestyle="--", color=roc_color, linewidth=max(1.0, roc_lw-1))
    _thicken_roc_lines(axes[0], lw=roc_lw, tick_fs=tick_fs, label_fs=label_fs, legend_fs=legend_fs, title_fs=title_fs)
    _place_letter_bottom(axes[0], letters[0], fontsize=letter_fs, pad_pts=letter_pad_pts)

    # Clients ROC
    for i, (ytc, ypc) in enumerate(client_pairs, start=0): # Start from 0
        ax_idx = i + 1
        y_true_to_use = ytc if ytc is not None else server_y_true
        _draw_multiclass_roc(axes[ax_idx], y_true_to_use, ypc, f"Client {i+1} ROC", class_labels=class_labels)
        if add_black_diagonal:
            axes[ax_idx].plot([0, 1], [0, 1], linestyle="--", color=roc_color, linewidth=max(1.0, roc_lw-1))
        _thicken_roc_lines(axes[ax_idx], lw=roc_lw, tick_fs=tick_fs, label_fs=label_fs, legend_fs=legend_fs, title_fs=title_fs)
        _place_letter_bottom(axes[ax_idx], letters[ax_idx if ax_idx < len(letters) else -1], fontsize=letter_fs, pad_pts=letter_pad_pts)

    suptitle = _compose_suptitle(
        title="ROC Curves",
        model_name=model_name,
        source_name=source_name,
        n_clients=n_clients,
        resource_label=resource_label
    )
    fig.suptitle(suptitle, fontsize=suptitle_fs, y=0.995)

    fig.tight_layout(rect=[0, 0.19, 1, 0.97])
    out_path = os.path.join(save_dir, filename or f"{source_name}_{model_name}_panel_ROC_server_and_clients_col.png")
    _assert_and_log_save(fig, out_path)
    print(f"[panel] ROC rows×cols: {nplots}×1  -> {out_path}")
    return out_path

# ===================================================================
# COMPOSITE PLOTS: GRID & ROW LAYOUTS (Horizontally Oriented)
# ===================================================================
def plot_grid_rocs(
    server_y_true, server_y_prob, client_pairs, save_dir, model_name, source_name,
    class_labels=None, dpi=180, filename=None,
    *,  # styling & scaling
    roc_lw=3.5, roc_color="black",
    tick_fs=13, label_fs=14, legend_fs=13, title_fs=18,
    letter_fs=13, letter_pad_pts=34, suptitle_fs=18,
    size_scale=3.00, fs_scale=3.00
):
    """
    Horizontal (grid) ROC panel: [Server | Client 1 | Client 2 | ...] in one row,
    styled like plot_panel_server_and_client_cms with the same suptitle bar.
    """
    n_clients = len(client_pairs)
    nplots = 1 + n_clients

    base_w_per_ax = 4.6
    base_h = 4.6
    fig_w = (base_w_per_ax * nplots) * size_scale
    fig_h = base_h * size_scale

    fig, axes = plt.subplots(nrows=1, ncols=nplots, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)

    # Scale font sizes
    ltfs = int(round(letter_fs * fs_scale))
    lpad = int(round(letter_pad_pts * fs_scale))
    stfs = int(round(suptitle_fs * fs_scale))
    tit_fs = int(round(title_fs * fs_scale))
    tk_fs  = int(round(tick_fs  * fs_scale))
    lb_fs  = int(round(label_fs * fs_scale))
    lg_fs  = int(round(legend_fs* fs_scale))
    rw_lw  = roc_lw

    # ---- Server ROC (col 0) ----
    _draw_multiclass_roc(axes[0], server_y_true, server_y_prob, "Server (Global) ROC", class_labels=class_labels)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=roc_color, linewidth=max(1.0, rw_lw-1))
    _thicken_roc_lines(axes[0], lw=rw_lw, tick_fs=tk_fs, label_fs=lb_fs, legend_fs=lg_fs, title_fs=tit_fs, roc_color=None)
    _place_letter_bottom(axes[0], letters[0], fontsize=ltfs, pad_pts=lpad)

    # ---- Client ROCs ----
    for i, (ytc, ypc) in enumerate(client_pairs, start=0): # Start from 0
        ax_idx = i + 1
        y_true_to_use = ytc if ytc is not None else server_y_true
        _draw_multiclass_roc(axes[ax_idx], y_true_to_use, ypc, f"Client {i+1} ROC", class_labels=class_labels)
        axes[ax_idx].plot([0, 1], [0, 1], linestyle="--", color=roc_color, linewidth=max(1.0, rw_lw-1))
        _thicken_roc_lines(axes[ax_idx], lw=rw_lw, tick_fs=tk_fs, label_fs=lb_fs, legend_fs=lg_fs, title_fs=tit_fs, roc_color=None)
        _place_letter_bottom(axes[ax_idx], letters[ax_idx if ax_idx < len(letters) else -1], fontsize=ltfs, pad_pts=lpad)

    fig.tight_layout(rect=[0, 0.22, 1, 0.83])
    fig.canvas.draw()
    left  = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    center_x = (left + right) / 2.0

    suptitle = _compose_suptitle(
        title="ROC Curves",
        model_name=model_name,
        source_name=source_name,
        n_clients=n_clients
    )
    fig.suptitle(suptitle, fontsize=stfs, y=0.995, x=center_x, ha="center")

    out = os.path.join(
        save_dir,
        filename or f"{source_name}_{model_name}_panel_ROC_server_and_clients_grid.png"
    )
    _assert_and_log_save(fig, out)
    return out

def plot_grid_cms(
    server_cm, client_cms, save_dir, model_name, source_name, class_labels=None,
    dpi=180, filename=None, *, cell_fs=16, cm_cmap="Blues",
    letter_fs=13, letter_pad_pts=34, suptitle_fs=18,
    size_scale=3.00, fs_scale=3.00
):
    """Horizontal (grid) CM panel: [Server | Client 1 | Client 2 | ...] in one row."""
    n_clients = len(client_cms)
    nplots = 1 + n_clients

    base_w_per_ax = 4.6
    base_h = 4.6
    fig_w = (base_w_per_ax * nplots) * size_scale
    fig_h = base_h * size_scale
    fig, axes = plt.subplots(nrows=1, ncols=nplots, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    axes = axes.flatten()
    letters = list(string.ascii_lowercase)

    # scaled sizes
    cfs  = int(round(cell_fs * fs_scale))
    ltfs = int(round(letter_fs * fs_scale))
    lpad = int(round(letter_pad_pts * fs_scale))
    stfs = int(round(suptitle_fs * fs_scale))
    title_fs_local = int(round(18 * fs_scale))
    label_fs_local = int(round(16 * fs_scale))
    tick_x_local   = int(round(13 * fs_scale))
    tick_y_local   = int(round(13 * fs_scale))
    cbar_fs_local  = int(round(14 * fs_scale))

    # Server
    _draw_cm_compat(
        axes[0], server_cm, "Server (Global)", class_labels=class_labels, cmap=cm_cmap,
        title_fs=title_fs_local, label_fs=label_fs_local,
        tick_fs_x=tick_x_local, tick_fs_y=tick_y_local, cbar_fs=cbar_fs_local
    )
    _bump_cm_cell_font(axes[0], anno_fs=cfs, color=None)
    _place_letter_bottom(axes[0], letters[0], fontsize=ltfs, pad_pts=lpad)

    # Clients
    for i, cm in enumerate(client_cms, start=0): # Start from 0
        ax_idx = i + 1
        _draw_cm_compat(
            axes[ax_idx], cm, f"Client {i+1}", class_labels=class_labels, cmap=cm_cmap,
            title_fs=title_fs_local, label_fs=label_fs_local,
            tick_fs_x=tick_x_local, tick_fs_y=tick_y_local, cbar_fs=cbar_fs_local
        )
        _bump_cm_cell_font(axes[ax_idx], anno_fs=cfs, color=None)
        _place_letter_bottom(axes[ax_idx], letters[ax_idx if ax_idx < len(letters) else -1], fontsize=ltfs, pad_pts=lpad)

    fig.tight_layout(rect=[0, 0.22, 1, 0.83])
    fig.canvas.draw()
    left  = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    center_x = (left + right) / 2.0

    suptitle = _compose_suptitle(
        title="Confusion Matrices",
        model_name=model_name,
        source_name=source_name,
        n_clients=n_clients
    )
    fig.suptitle(suptitle, fontsize=stfs, y=0.995, x=center_x, ha="center")

    out = os.path.join(save_dir, filename or f"{source_name}_{model_name}_panel_confusion_matrices.png")
    _assert_and_log_save(fig, out)
    return out

def plot_row_cms_panel(
    experiments, save_dir, dpi=180, filename=None,
    *, cell_fs=16, cm_cmap="Blues", letter_fs=13, letter_pad_pts=34, suptitle_fs=18,
    size_scale=2.25, fs_scale=2.25
):
    """Stacks multiple experiment *rows* vertically."""
    max_nplots = max(1 + len(e.get("client_cms", [])) for e in experiments)
    nrows = len(experiments)

    base_w_per_ax = 4.6
    base_h = 4.6
    fig_w = (base_w_per_ax * max_nplots) * size_scale
    fig_h = (base_h * nrows) * size_scale

    fig, axes = plt.subplots(nrows=nrows, ncols=max_nplots, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    letters = list(string.ascii_lowercase)
    L = 0

    cfs  = int(round(cell_fs * fs_scale))
    ltfs = int(round(letter_fs * fs_scale))
    lpad = int(round(letter_pad_pts * fs_scale))
    stfs = int(round(suptitle_fs * fs_scale))

    for r, exp in enumerate(experiments):
        server_cm   = exp["server_cm"]
        client_cms  = list(exp.get("client_cms", []))
        class_labels= exp.get("class_labels")
        n_clients   = len(client_cms)

        _draw_cm_compat(axes[r,0], server_cm, "Server (Global)", class_labels=class_labels, cmap=cm_cmap)
        _bump_cm_cell_font(axes[r,0], anno_fs=cfs, color=None)
        _place_letter_bottom(axes[r,0], letters[L if L < len(letters) else -1], fontsize=ltfs, pad_pts=lpad); L += 1

        for i, cm in enumerate(client_cms, start=0): # Start from 0
            ax_idx = i + 1
            _draw_cm_compat(axes[r,ax_idx], cm, f"Client {i+1}", class_labels=class_labels, cmap=cm_cmap)
            _bump_cm_cell_font(axes[r,ax_idx], anno_fs=cfs, color=None)
            _place_letter_bottom(axes[r,ax_idx], letters[L if L < len(letters) else -1], fontsize=ltfs, pad_pts=lpad); L += 1

        for j in range(1 + n_clients, max_nplots):
            axes[r,j].set_axis_off()

    first = experiments[0]
    suptitle = _compose_suptitle(
        title="Confusion Matrices",
        model_name=first.get("model_name","Model"),
        source_name=first.get("source_name","Dataset"),
        n_clients=len(first.get("client_cms",[]))
    )
    fig.suptitle(suptitle, fontsize=stfs, y=0.995)

    fig.tight_layout(rect=[0, 0.22, 1, 0.83])
    out = os.path.join(save_dir, filename or f"{first.get('source_name','Dataset')}_{first.get('model_name','Model')}_panel_confusion_matrices_rows.png")
    _assert_and_log_save(fig, out)
    return out

def plot_row_rocs_panel(
    experiments, save_dir, dpi=180, filename=None,
    *, roc_lw=3.5, roc_color="black", title_fs=18, tick_fs=13, label_fs=14, legend_fs=13,
    letter_fs=13, letter_pad_pts=34, suptitle_fs=18
):
    """Stacks multiple experiment *rows* vertically for ROCs."""
    max_nplots = max(1 + len(e.get("client_pairs", [])) for e in experiments)
    nrows = len(experiments)
    fig_w = 4.8 * max_nplots
    fig_h = 4.8 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=max_nplots, figsize=(fig_w, fig_h), dpi=dpi, squeeze=False)
    letters = list(string.ascii_lowercase)
    L = 0

    for r, exp in enumerate(experiments):
        server_y_true = exp["server_y_true"]
        server_y_prob = exp["server_y_prob"]
        client_pairs  = list(exp.get("client_pairs", []))
        class_labels  = exp.get("class_labels")
        n_clients = len(client_pairs)

        _draw_multiclass_roc(axes[r,0], server_y_true, server_y_prob, "Server (Global) ROC", class_labels=class_labels)
        axes[r,0].plot([0,1],[0,1], linestyle="--", color=roc_color, linewidth=max(1.0, roc_lw-1))
        _thicken_roc_lines(axes[r,0], lw=roc_lw, tick_fs=tick_fs, label_fs=label_fs, legend_fs=legend_fs, title_fs=title_fs)
        _place_letter_bottom(axes[r,0], letters[L if L < len(letters) else -1], fontsize=letter_fs, pad_pts=letter_pad_pts); L += 1

        for i, (ytc, ypc) in enumerate(client_pairs, start=0): # Start from 0
            ax_idx = i + 1
            y_true_to_use = ytc if ytc is not None else server_y_true
            _draw_multiclass_roc(axes[r,ax_idx], y_true_to_use, ypc, f"Client {i+1} ROC", class_labels=class_labels)
            axes[r,ax_idx].plot([0,1],[0,1], linestyle="--", color=roc_color, linewidth=max(1.0, roc_lw-1))
            _thicken_roc_lines(axes[r,ax_idx], lw=roc_lw, tick_fs=tick_fs, label_fs=label_fs, legend_fs=legend_fs, title_fs=title_fs)
            _place_letter_bottom(axes[r,ax_idx], letters[L if L < len(letters) else -1], fontsize=letter_fs, pad_pts=letter_pad_pts); L += 1

        for j in range(1 + n_clients, max_nplots):
            axes[r,j].set_axis_off()

    first = experiments[0]
    suptitle = _compose_suptitle(
        title="ROC Curves",
        model_name=first.get("model_name","Model"),
        source_name=first.get("source_name","Dataset"),
        n_clients=len(first.get("client_pairs",[]))
    )
    fig.suptitle(suptitle, fontsize=suptitle_fs, y=0.995)
    fig.tight_layout(rect=[0, 0.22, 1, 0.83])
    out = os.path.join(save_dir, filename or f"{first.get('source_name','Dataset')}_{first.get('model_name','Model')}_panel_ROC_server_and_clients_rows.png")
    _assert_and_log_save(fig, out)
    return out

