import os
import tarfile
import urllib.request
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
from pydicom import dcmread
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import BayesianGaussianMixture


# ==========================================
# 1. DATASET DEFINITION (DUAL PNG/DCM LOADER)
# ==========================================
class IndustrialAnomalyDataset(Dataset):
    def __init__(self, base_path, target_size=448):
        self.base_path = Path(base_path)
        self.target_size = target_size
        self.filepaths = []
        self.labels = []

        # Recursively search for DICOM or PNG files
        for f in self.base_path.rglob("*"):
            if f.is_file() and f.suffix.upper() in [".DCM", ".PNG", ".JPG", ".JPEG"]:
                self.filepaths.append(f)
                relative_path = f.relative_to(self.base_path)
                # Label is the top-level folder name (e.g., 'good', 'broken_grid')
                self.labels.append(relative_path.parts[0])

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.filepaths)

    def preprocess_image(self, img_raw, is_dcm=False):
        """
        Dynamically applies robust percentile scaling to raw DCM 16-bit files,
        or handles standard 8-bit images cleanly without warping geometry.
        """
        img_float = img_raw.astype(np.float32)

        if is_dcm:
            # 16-bit raw industrial X-ray: scale compressed dynamic range
            p_low, p_high = np.percentile(img_float, (1.0, 99.0))
            if p_high <= p_low:
                p_high = p_low + 1e-6
            image = np.clip(img_float, p_low, p_high)
            image = (image - p_low) / (p_high - p_low)
        else:
            # Standard 8-bit image: scale to [0, 1]
            image = img_float / 255.0

        # Aspect-Preserved Isotropic Resizing (Letterboxing)
        ch, cw = image.shape[:2]
        scale = self.target_size / max(ch, cw)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Inject into zero-padded canvas (keeps repeating pattern geometry accurate!)
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        x_offset = (self.target_size - new_w) // 2
        y_offset = (self.target_size - new_h) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Create 3-channel tensor [3, H, W]
        img_tensor = torch.from_numpy(canvas).unsqueeze(0).repeat(3, 1, 1)
        img_tensor = self.normalize(img_tensor)

        return img_tensor, canvas

    def __getitem__(self, index):
        path = self.filepaths[index]
        label = self.labels[index]
        is_dcm = path.suffix.upper() == ".DCM"

        if is_dcm:
            ds = dcmread(path)
            img_raw = ds.pixel_array
        else:
            img_raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                raise RuntimeError(f"Failed to read image at: {path}")

        img_tensor, processed_canvas = self.preprocess_image(img_raw, is_dcm=is_dcm)
        return img_tensor, label, str(path), processed_canvas


# ==========================================
# 2. DATASET DOWNLOADER (MVTEC GRID PROXY)
# ==========================================
def download_mvtec_grid():
    """Downloads and extracts MVTec Grid category to act as your repeating pattern proxy"""
    dataset_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/grid.tar.xz"
    archive_path = "grid.tar.xz"
    extract_dir = "dataset"

    if not os.path.exists(os.path.join(extract_dir, "grid")):
        os.makedirs(extract_dir, exist_ok=True)
        print("📥 Downloading MVTec Grid dataset (repeating patterns)...")
        urllib.request.urlretrieve(dataset_url, archive_path)
        print("📦 Extracting dataset archives...")
        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(path=extract_dir)
        print("✅ Dataset loaded successfully!")
    else:
        print("📂 MVTec Grid dataset directory already exists.")


# ==========================================
# 3. PIPELINE RUNNER
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using execution device: {device}")

    # Set up folders
    os.makedirs("figures", exist_ok=True)
    download_mvtec_grid()

    # Step 1: Initialize Datasets
    # Grid patterns have spacing parameters analogous to approximately 5 mm spacing
    train_path = "dataset/grid/train"
    test_path = "dataset/grid/test"

    train_dataset = IndustrialAnomalyDataset(train_path, target_size=448)
    test_dataset = IndustrialAnomalyDataset(test_path, target_size=448)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Step 2: Load local or standard DINOv2 Backbone
    print("🤖 Loading pre-trained DINOv2 Backbone (ViT-S/14)...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2 = dinov2.to(device).eval()

    # Step 3: Extract normal representations for Dirichlet Clustering
    print("🧬 Extracting normal features from the golden baseline...")
    all_features = []

    with torch.no_grad():
        for imgs, _, _, _ in train_loader:
            imgs = imgs.to(device)
            # Pull intermediate layers (layer 11 is highly responsive to repeating structures)
            features = dinov2.get_intermediate_layers(imgs, n=1)[0]
            # Shape: [Batch, Patches (1024), Dim (384)]
            features = features.reshape(
                -1, features.shape[-1]
            )  # Flatten to [N_patches, Dim]

            # L2 Normalize features for robust cosine metric alignment
            features = F.normalize(features, p=2, dim=-1)
            all_features.append(features.cpu().numpy())

    normal_features = np.concatenate(all_features, axis=0)
    print(
        f"📊 Collected {normal_features.shape[0]} baseline patches of dimension {normal_features.shape[1]}"
    )

    # Step 4: Fit variational DP-GMM (No manual K selection needed!)
    # Scikit-learn's BayesianGaussianMixture implements the Dirichlet Process prior natively
    print("🔮 Clustering features using Variational Dirichlet Process Mixture Model...")
    dpmm = BayesianGaussianMixture(
        n_components=150,  # Max allowed clusters; DPMM will suppress unused ones
        covariance_type="diag",  # 'diag' is fast, efficient, and avoids singular matrices
        weight_concentration_prior=1e-3,
        max_iter=100,
        random_state=42,
    )
    dpmm.fit(normal_features)

    # Filter for active components with substantial weight support
    active_weights = dpmm.weights_ > 0.01
    centroids = dpmm.means_[active_weights]
    print(
        f"✨ Dirichlet Process converged. Compressed feature space to {len(centroids)} active pattern clusters."
    )

    # Convert centroids back to PyTorch for blazing-fast GPU matrix distances
    centroids_tensor = torch.from_numpy(centroids).to(device)  # Shape: [K_active, Dim]

    # Step 5: Test and Visualize Anomaly Localization
    print("🔍 Running test evaluation on anomalous patterns...")
    test_limit = 5
    visualized_count = 0

    with torch.no_grad():
        for imgs, labels, paths, canvases in test_loader:
            if visualized_count >= test_limit:
                break

            label = labels[0]
            if label == "good":  # Let's target anomalous classes specifically
                continue

            path = paths[0]
            canvas = canvases[0].numpy()
            imgs = imgs.to(device)

            # Extract query features
            test_features = dinov2.get_intermediate_layers(imgs, n=1)[0].squeeze(
                0
            )  # [1024, Dim]
            test_features = F.normalize(test_features, p=2, dim=-1)  # [1024, Dim]

            # Calculate cosine distances to nearest cluster centroid
            # Cosine distance = 1.0 - Cosine Similarity
            cos_similarity = torch.matmul(
                test_features, centroids_tensor.T
            )  # [1024, K_active]
            cos_distance = 1.0 - cos_similarity

            # Anomaly score per patch is the distance to its closest normal match cluster
            min_distances, _ = torch.min(cos_distance, dim=-1)  # [1024]
            image_anomaly_score = torch.max(min_distances).item()

            # Reshape back to spatial grid (448 / 14 = 32 patches per side)
            grid_dim = 448 // 14
            anomaly_map = min_distances.reshape(1, 1, grid_dim, grid_dim)

            # Interpolate to input size [448, 448]
            anomaly_map = (
                F.interpolate(
                    anomaly_map, size=(448, 448), mode="bicubic", align_corners=False
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # Soft Gaussian blur to smooth grid boundaries
            anomaly_map = cv2.GaussianBlur(anomaly_map, (15, 15), 0)

            # Plot visual report
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(canvas, cmap="gray")
            ax1.set_title("Input Image")
            ax1.axis("off")

            # Show Jet colormap overlay of the anomalies
            ax2.imshow(anomaly_map, cmap="jet")
            ax2.set_title(f"Anomaly Map (Score: {image_anomaly_score:.3f})")
            ax2.axis("off")

            # Combined alpha overlay
            ax3.imshow(canvas, cmap="gray")
            ax3.imshow(anomaly_map, cmap="jet", alpha=0.5)
            ax3.set_title(f"Overlay | Type: {label}")
            ax3.axis("off")

            fig.suptitle(f"File: {os.path.basename(path)}", fontsize=12)
            plt.tight_layout()

            output_fig_path = f"figures/defect_{visualized_count}.png"
            plt.savefig(output_fig_path, dpi=150)
            print(f"💾 Saved anomaly visual analysis report to: {output_fig_path}")
            plt.close()

            visualized_count += 1


if __name__ == "__main__":
    main()
