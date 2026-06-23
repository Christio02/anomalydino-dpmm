import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.mixture import BayesianGaussianMixture
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

# Import the dataset from the main script
# Ensure run_pipeline.py is importable
try:
    from run_pipeline import IndustrialAnomalyDataset
except ImportError:
    print("❌ Failed to import IndustrialAnomalyDataset from run_pipeline.py")
    exit(1)


def run_audits(data_root):
    print("🔍 Starting DINOv2 + DP-GMM Pipeline Verification...\n")

    # Check 1: Device Assignment
    print("[Check 1] Device Assignment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(
            "  ⚠️  WARNING: CUDA is not available. Running on CPU. This will be slow!"
        )
    else:
        print("  ✅ CUDA is available. Using GPU.")

    # Check 2: Data Loading & Directory Layout
    print("\n[Check 2] Data Layout & Loader...")
    train_path = os.path.join(data_root, "train")
    if not os.path.exists(os.path.join(train_path, "good")):
        print(
            f"  ❌ FAILED: Expected path {os.path.join(train_path, 'good')} does not exist."
        )
        print(
            f"     Please ensure data is structured exactly as: {data_root}/train/good/"
        )
        exit(1)

    train_dataset = IndustrialAnomalyDataset(train_path, target_size=448)
    EXPECTED_COUNT = 264
    if len(train_dataset) != EXPECTED_COUNT:
        print(
            f"  ⚠️  WARNING: Expected {EXPECTED_COUNT} samples (MVTec Grid), found {len(train_dataset)}. Continuing..."
        )

    labels_set = set(train_dataset.labels)
    if labels_set != {"good"}:
        print(f"  ❌ FAILED: Expected labels to only be 'good', found {labels_set}.")
        exit(1)
    print(
        f"  ✅ Data loaded correctly. N={len(train_dataset)} samples, Labels={labels_set}."
    )

    # Check 3 & 4: Tensor Math & Bounding Range
    print("\n[Check 3 & 4] Tensor Math & Bounding Range...")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    # Grab one batch
    img_tensors, labels, paths, canvases = next(iter(train_loader))

    # 3. Assert Shape
    if img_tensors.shape != (8, 3, 448, 448):
        print(
            f"  ❌ FAILED: Batched tensor shape mismatch! Expected (8, 3, 448, 448), got {tuple(img_tensors.shape)}"
        )
        exit(1)

    # 4. Assert Data Range
    min_val, max_val = img_tensors.min().item(), img_tensors.max().item()
    print(f"  Info: Normalized tensor bounds -> Min: {min_val:.3f}, Max: {max_val:.3f}")
    if not (-2.5 <= min_val <= -1.5) or not (1.5 <= max_val <= 3.0):
        print(
            f"  ❌ FAILED: Tensor normalization looks wrong. Min: {min_val:.3f}, Max: {max_val:.3f}"
        )
        exit(1)
    print("  ✅ Tensor dimensions and normalization bounds verified.")

    # Check 5: Feature Extraction Shapes
    print("\n[Check 5] DINOv2 Feature Extraction Shapes...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2 = dinov2.to(device).eval()

    with torch.no_grad():
        img_tensors_device = img_tensors.to(device)
        features = dinov2.get_intermediate_layers(img_tensors_device, n=1)[0]

        # Batch: 8, Patches: 32x32 = 1024, Embedding: 384
        if features.shape != (8, 1024, 384):
            print(
                f"  ❌ FAILED: Raw features shape expected (8, 1024, 384), got {tuple(features.shape)}"
            )
            exit(1)

        # Flatten and L2 Normalize
        features_flat = features.reshape(-1, 384)
        features_norm = F.normalize(features_flat, p=2, dim=-1)

        expected_shape = (8192, 384)
        if features_norm.shape != expected_shape:
            print(
                f"  ❌ FAILED: Normalized flat shape expected {expected_shape}, got {tuple(features_norm.shape)}"
            )
            exit(1)
    print(
        f"  ✅ DINOv2 accurately extracted and normalized shape: {tuple(features_norm.shape)}"
    )

    # Check 6: Clustering Convergence
    print("\n[Check 6] DP-GMM Clustering Convergence...")
    features_np = features_norm.cpu().numpy()
    gmm = BayesianGaussianMixture(
        n_components=150,
        covariance_type="diag",
        weight_concentration_prior=1e-3,
        max_iter=50,
        random_state=42,
    )
    gmm.fit(features_np)

    active_mask = gmm.weights_ > 0.01
    active_centroids = gmm.means_[active_mask]
    k_active = active_centroids.shape[0]

    print(f"  Info: Variational inference kept {k_active}/150 components active.")
    if k_active >= 150:
        print(
            "  ❌ FAILED: DP-GMM failed to compress/threshold the feature space (found >= 150 centroids)."
        )
        exit(1)
    elif k_active == 0:
        print("  ❌ FAILED: DP-GMM failed to fit properly (0 centroids active).")
        exit(1)
    print(
        f"  ✅ Bayesian clustering successfully converged down to {k_active} active centroids."
    )

    # Check 7: Plot Formatting and E2E Simulation
    print("\n[Check 7] E2E Visualization Saving Output...")
    os.makedirs("figures", exist_ok=True)

    # Mocking anomaly scoring
    mock_query = features_np[:1024]  # first image in batch
    dist = 1 - np.dot(mock_query, active_centroids.T)
    min_dist = np.min(dist, axis=-1)

    heatmap = min_dist.reshape(32, 32)
    heatmap_resized = cv2.resize(heatmap, (448, 448), interpolation=cv2.INTER_CUBIC)
    heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)

    fig, ax = plt.subplots()
    ax.imshow(canvases[0], cmap="gray")
    ax.imshow(heatmap_blurred, cmap="jet", alpha=0.5)

    output_path = "figures/verify_pipeline_test.png"
    plt.savefig(output_path)
    plt.close()

    if not os.path.exists(output_path):
        print("  ❌ FAILED: Mock pipeline failed to write to figures/ directory.")
        exit(1)

    print(f"  ✅ Localisation output plot successfully created at {output_path}")
    print("\n🎉 ALL PIPELINE VERIFICATION CHECKS PASSED. PIPELINE IS HEALTHY! 🎉")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify anomaly detection pipeline setup."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="dataset/grid",
        help="Root directory of the dataset",
    )
    args = parser.parse_args()
    run_audits(args.data_root)
