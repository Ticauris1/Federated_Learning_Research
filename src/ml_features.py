
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

def _infer_square_side(d: int) -> int:
        """Infers the side length of a square image from its flattened dimension."""
        s = int(round((d / 3.0) ** 0.5))
        if 3 * s * s != d:
            s = int(np.sqrt(d // 3))
        return max(s, 1)

def hogify_flat(X):
        """Flattened RGB in [0,1] -> grayscale HOG features per sample."""
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        side = _infer_square_side(d)
        imgs = X.reshape(n, 3, side, side).mean(axis=1)  # grayscale
        feats = [
            hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                block_norm='L2-Hys', feature_vector=True)
            for img in imgs
        ]
        return np.asarray(feats, dtype=np.float32)

def color_hist_flat(X, bins=32):
        """Flattened RGB in [0,1] -> per-channel color histograms (3*bins)."""
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        side = _infer_square_side(d)
        imgs = X.reshape(n, 3, side, side)
        hists = []
        for img in imgs:
            chans = [np.histogram(img[c].ravel(), bins=bins, range=(0, 1), density=True)[0] for c in range(3)]
            hists.append(np.concatenate(chans))
        return np.asarray(hists, dtype=np.float32)

    # Export transformers and the union
HOG   = ("hog",   FunctionTransformer(hogify_flat, validate=False))
COLOR = ("color", FunctionTransformer(color_hist_flat, kw_args={"bins": 32}, validate=False))
FEATS = ("feats", FeatureUnion([HOG, COLOR]))
    
print("✅ utils.py created successfully.")