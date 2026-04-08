import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageOps
from scipy.ndimage import sobel
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────

CROP_SIZE    = 50
N_COMPONENTS = 30
DATASET_DIR  = "dataset"
TEST_DIR     = "test"

CLASSES = ["disease", "healthy", "nitrogen_deficiency", "water_stress"]

CLASS_LABELS = {
    "disease": "🔴 Diseased",
    "healthy": "🟢 Healthy",
    "nitrogen_deficiency": "🟡 Nitrogen Deficient",
    "water_stress": "🔵 Water Stress",
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ──────────────────────────────
# FLAGS (to avoid repeated prints)
# ──────────────────────────────

printed_flags = {
    "matrix_representation": False,
    "rref": False,
    "orthogonalization": False,
    "rank_nullity": False,
    "eigen": False,
    "diagonalization": False,
    "least_squares": False,
    "projection": False,
    "final_output": False,
}

# ──────────────────────────────
# IMAGE PREPROCESSING + FEATURES
# ──────────────────────────────

def centre_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    if w < size or h < size:
        img = img.resize((max(w, size), max(h, size)), Image.LANCZOS)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))

def load_dataset(dataset_dir: str, classes: list, crop_size: int):
    X, y, paths = [], [], []
    for idx, name in enumerate(classes):
        folder = os.path.join(dataset_dir, name)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
        print(f"  Loading {len(files)} images  ←  {name}/")
        for f in files:
            try:
                X.append(extract_features(os.path.join(folder, f), crop_size))
                y.append(idx)
                paths.append(os.path.join(folder, f))
            except:
                continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths

def load_test_images(test_dir: str, crop_size: int):
    X_test, paths = [], []
    files = sorted([f for f in os.listdir(test_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS])
    for f in files:
        try:
            X_test.append(extract_features(os.path.join(test_dir, f), crop_size))
            paths.append(os.path.join(test_dir, f))
        except:
            continue
    return np.array(X_test, dtype=np.float32), paths

def extract_features(img_path: str, crop_size: int) -> np.ndarray:
    if not printed_flags["matrix_representation"]:
        print(" REAL-WORLD DATA → Matrix Representation executed")
        printed_flags["matrix_representation"] = True

    img = Image.open(img_path).convert("RGB")
    crop = centre_crop(img, crop_size)
    
    rgb = np.array(crop, dtype=np.float32).flatten() / 255.0
    
    gray = np.array(ImageOps.grayscale(crop), dtype=np.float32) / 255.0
    gray_flat = gray.flatten()
    
    sobel_x = sobel(gray, axis=0)
    sobel_y = sobel(gray, axis=1)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_flat = (sobel_mag / (sobel_mag.max() + 1e-8)).flatten()
    
    feature_vec = np.concatenate([rgb, gray_flat, sobel_flat])
    return feature_vec

# ──────────────────────────────
# GAUSSIAN ELIMINATION + ORTHOGONAL BASIS
# ──────────────────────────────

def get_rref_basis(A, k, tol=1e-8):
    if not printed_flags["rref"]:
        print(" Matrix Simplification (RREF) executed")
        printed_flags["rref"] = True

    matrix = A.copy()
    rows, cols = matrix.shape
    pivot_row = 0
    basis_indices = []
    for j in range(cols):
        if pivot_row >= rows or len(basis_indices) >= k: break
        max_row = np.argmax(np.abs(matrix[pivot_row:, j])) + pivot_row
        if np.abs(matrix[max_row, j]) < tol: continue
        matrix[[pivot_row, max_row]] = matrix[[max_row, pivot_row]]
        lv = matrix[pivot_row, j]
        matrix[pivot_row] /= lv
        for i in range(rows):
            if i != pivot_row:
                matrix[i] -= matrix[i, j] * matrix[pivot_row]
        basis_indices.append(pivot_row)
        pivot_row += 1
    return A[basis_indices]

def gram_schmidt(B):
    if not printed_flags["orthogonalization"]:
        print(" Orthogonalization (Gram-Schmidt) executed")
        printed_flags["orthogonalization"] = True

    ortho_basis = []
    for v in B:
        for u in ortho_basis:
            v -= np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            ortho_basis.append(v / norm)
    return np.array(ortho_basis)

# ──────────────────────────────
# EIGENVECTORS 
# ──────────────────────────────

def analyze_matrix_properties(A, name="Matrix"):
    print(f"\n--- {name} Analysis ---")
    
    if not printed_flags["rank_nullity"]:
        print(" Structure of the Space (Rank & Nullity) executed")
        printed_flags["rank_nullity"] = True

    rank = np.linalg.matrix_rank(A)
    nullity = A.shape[1] - rank
    
    print(f"Rank: {rank}")
    print(f"Nullity: {nullity}")
    print(f"(Rank + Nullity = {rank + nullity} = columns {A.shape[1]})")

    try:
        if not printed_flags["eigen"]:
            print(" Pattern Discovery (Eigenvalues & Eigenvectors) executed")
            printed_flags["eigen"] = True

        A_small = A[:, :200] if A.shape[1] > 200 else A
        
        cov = A_small.T @ A_small
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        print(f"Top Eigenvalues: {eigenvalues[:5]}")

        if not printed_flags["diagonalization"]:
            print(" System Simplification (Diagonalization) executed")
            printed_flags["diagonalization"] = True

        P = eigenvectors
        D = np.diag(eigenvalues)
        P_inv = np.linalg.inv(P)
        reconstructed = P @ D @ P_inv

        print("Diagonalization approx valid:",
              np.allclose(cov, reconstructed, atol=1e-5))
    except:
        print("Eigen/Diagonalization skipped")

def least_squares_projection(x, basis):
    if not printed_flags["least_squares"]:
        print(" Prediction / Approximation (Least Squares) executed")
        printed_flags["least_squares"] = True

    B = basis.T
    try:
        coeffs = np.linalg.pinv(B) @ x
        approx = B @ coeffs
        return approx
    except:
        return np.zeros_like(x)

# ──────────────────────────────
# PROJECTION + CLASSIFICATION
# ──────────────────────────────

def project_onto_subspace(vec, basis):
    if not printed_flags["projection"]:
        print(" Projection onto Subspace executed")
        printed_flags["projection"] = True

    proj = np.zeros_like(vec)
    for u in basis:
        proj += np.dot(vec, u) * u
    return proj

def classify_leaves(X_test, class_bases):
    preds, distances = [], []
    for x in X_test:
        errs = []
        for basis in class_bases:
            proj = project_onto_subspace(x, basis)

            _ = least_squares_projection(x, basis)

            errs.append(np.linalg.norm(x - proj))
        preds.append(np.argmin(errs))
        distances.append(errs)

    if not printed_flags["final_output"]:
        print(" FINAL APPLICATION OUTPUT (Predictions) executed")
        printed_flags["final_output"] = True

    return np.array(preds), np.array(distances)

# ──────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────

def main():
    print("=" * 62)
    print("  LEAF CLASSIFICATION  ·  Linear Algebra + Texture Features")
    print("=" * 62)

    X_train, y_train, _ = load_dataset(DATASET_DIR, CLASSES, CROP_SIZE)

    analyze_matrix_properties(X_train, "Training Data Matrix")

    class_bases = []
    for i, c in enumerate(CLASSES):
        class_data = X_train[y_train == i]

        analyze_matrix_properties(class_data, f"{c} Class Matrix")

        raw_basis = get_rref_basis(class_data, N_COMPONENTS)
        print(" Remove Redundancy (Basis Selection) executed")

        ortho_basis = gram_schmidt(raw_basis)
        class_bases.append(ortho_basis)
        print(f"Basis for {c:20} formed with rank {len(ortho_basis)}")

    X_test, test_paths = load_test_images(TEST_DIR, CROP_SIZE)

    predictions, distances = classify_leaves(X_test, class_bases)

    print("\n" + "─"*62)
    print(f"{'FILE':<20}  {'PREDICTED CLASS':<30}")
    print("─"*62)
    for i, f in enumerate(test_paths):
        pred_name = CLASSES[predictions[i]]
        print(f"{os.path.basename(f):<20}  {CLASS_LABELS[pred_name]:<30}")
    print("─"*62)


if __name__ == "__main__":
    main()