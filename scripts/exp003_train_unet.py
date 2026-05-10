"""
Exp003: 基础分割 Baseline — U-Net 训练

目标：在 150 例 Training/Labeled 上训练 2D U-Net
验证：训练 loss 收敛，输出预测对比图

流程：
  1. 遍历全部 150 例
  2. 每例选 ED frame，过预处理 pipeline → [1, 256, 256, Z]
  3. 提取有标签的 Z-slice 作为 2D 训练样本
  4. 80% 训练 / 20% 验证
  5. 训练 MONAI BasicUNet（Dice Loss）
  6. 输出 loss 曲线 + 预测 overlay
"""

from pathlib import Path
import csv
import random
from collections import defaultdict

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
)
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "exp003"
CSV_PATH = DATASET_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.benchmark = True
    USE_AMP = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    USE_AMP = False
else:
    DEVICE = "cpu"
    USE_AMP = False


# ============================================================
# 1. 选 ED frame（复用）
# ============================================================

def select_annotated_frame(image_data, label_data):
    if image_data.ndim == 4:
        for t in range(label_data.shape[-1]):
            if np.any(label_data[..., t] > 0):
                return image_data[..., t], label_data[..., t], t
        return image_data[..., 0], label_data[..., 0], 0
    return image_data, label_data, -1


# ============================================================
# 2. 预处理 pipeline（复用 Exp002）
# ============================================================

def build_pipeline():
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5),
                 mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image",
                        select_fn=lambda x: x > 0),
        SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, -1)),
    ])


# ============================================================
# 3. 加载 & 预处理全部数据
# ============================================================

def load_and_preprocess_all(case_ids, pipeline):
    """对每个病例：加载 NIfTI → 选 ED frame → pipeline → 提取有标签的 2D slices"""
    all_images = []  # 每个元素: [1, 256, 256]
    all_labels = []  # 每个元素: [256, 256]
    all_meta = []    # 每个元素: {case_id, vendor}

    for case_id in tqdm(case_ids, desc="  预处理", unit="例"):
        image_path = DATASET_ROOT / "Training" / "Labeled" / case_id / f"{case_id}_sa.nii.gz"
        label_path = DATASET_ROOT / "Training" / "Labeled" / case_id / f"{case_id}_sa_gt.nii.gz"

        if not image_path.exists():
            continue

        # 加载
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))
        image_4d = image_nii.get_fdata()
        label_4d = label_nii.get_fdata()

        # 选 ED frame
        image_3d, label_3d, _ = select_annotated_frame(image_4d, label_4d)

        # 预处理 pipeline
        output = pipeline({"image": image_3d, "label": label_3d})
        vol_img = output["image"].numpy()
        vol_lbl = output["label"].numpy()

        # 逐层提取非空白 slice
        for z in range(vol_lbl.shape[-1]):
            lbl_slice = vol_lbl[0, :, :, z]
            if np.sum(lbl_slice > 0) > 20:
                all_images.append(vol_img[:, :, :, z].copy())
                all_labels.append(lbl_slice[None, :, :].copy())
                all_meta.append(case_id)

    print(f"  共 {len(all_images)} 个 2D slice（来自 {len(case_ids)} 例）")
    return all_images, all_labels, all_meta


# ============================================================
# 4. 训练
# ============================================================

def train(model, train_loader, val_loader, epochs, val_dataset=None):
    import time
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    history = {"train_loss": [], "val_loss": []}

    epoch_iter = tqdm(range(epochs), desc="Epoch")

    for epoch in epoch_iter:
        epoch_start = time.time()
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc="  Train", leave=False):
            x = batch["image"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(DEVICE)
                y = batch["label"].to(DEVICE)
                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                        loss = loss_fn(logits, y)
                else:
                    logits = model(x)
                    loss = loss_fn(logits, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        epoch_time = time.time() - epoch_start

        epoch_iter.set_postfix({
            "train": f"{avg_train_loss:.4f}",
            "val": f"{avg_val_loss:.4f}",
            "time": f"{epoch_time:.0f}s",
        })

    return history


# ============================================================
# 5. 预测可视化
# ============================================================

def visualize_predictions(model, val_loader, save_path, num_samples=4):
    model.eval()
    batch = next(iter(val_loader))
    x = batch["image"][:num_samples].to(DEVICE)
    y_true = batch["label"][:num_samples].numpy()

    with torch.no_grad():
        logits = model(x)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()  # [B, 256, 256]

    x_cpu = x.cpu().numpy()

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    for i in range(num_samples):
        # MRI
        axes[i, 0].imshow(x_cpu[i, 0], cmap="gray")
        axes[i, 0].set_title("MRI")
        axes[i, 0].axis("off")

        # Ground truth
        axes[i, 1].imshow(y_true[i, 0], cmap="viridis", vmin=0, vmax=3)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        # Prediction
        axes[i, 2].imshow(x_cpu[i, 0], cmap="gray")
        axes[i, 2].imshow(y_pred[i], cmap="jet", alpha=0.4, vmin=0, vmax=3)
        axes[i, 2].set_title("Prediction (overlay)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"预测图保存: {save_path}")


# ============================================================
# 6. Loss 曲线
# ============================================================

def plot_loss(history, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss 曲线保存: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Exp003: 基础分割 Baseline 训练")
    print(f"使用设备: {DEVICE} (AMP={'ON' if USE_AMP else 'OFF'})")
    print("=" * 60)

    # 读取 CSV，获取 Training/Labeled 的 case list
    case_ids = sorted([
        d.name for d in (DATASET_ROOT / "Training" / "Labeled").iterdir()
        if d.is_dir()
    ])
    print(f"\nTraining/Labeled 病例数: {len(case_ids)}")

    # 按 vendor 划分：80% 训练 / 20% 验证（保证 vendor 比例一致）
    random.seed(42)

    # 获取每个 case 的 vendor
    case_vendor = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["External code"] in case_ids:
                case_vendor[r["External code"]] = r["VendorName"]

    train_ids, val_ids = [], []
    for vendor in ["Siemens", "Philips"]:
        v_cases = [c for c in case_ids if case_vendor.get(c) == vendor]
        random.shuffle(v_cases)
        split = int(len(v_cases) * 0.8)
        train_ids.extend(v_cases[:split])
        val_ids.extend(v_cases[split:])

    print(f"  训练: {len(train_ids)} 例 (Siemens={sum(1 for c in train_ids if case_vendor[c]=='Siemens')}, "
          f"Philips={sum(1 for c in train_ids if case_vendor[c]=='Philips')})")
    print(f"  验证: {len(val_ids)} 例 (Siemens={sum(1 for c in val_ids if case_vendor[c]=='Siemens')}, "
          f"Philips={sum(1 for c in val_ids if case_vendor[c]=='Philips')})")

    # 预处理
    print("\n构建预处理 pipeline...")
    pipeline = build_pipeline()

    print("\n预处理训练数据...")
    train_imgs, train_lbls, _ = load_and_preprocess_all(train_ids, pipeline)

    print("\n预处理验证数据...")
    val_imgs, val_lbls, _ = load_and_preprocess_all(val_ids, pipeline)

    # 创建 Dataset & DataLoader
    train_dataset = Dataset(
        data=[{"image": img, "label": lbl} for img, lbl in zip(train_imgs, train_lbls)],
        transform=None,
    )
    val_dataset = Dataset(
        data=[{"image": img, "label": lbl} for img, lbl in zip(val_imgs, val_lbls)],
        transform=None,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 创建模型
    print("\n创建 U-Net...")
    model = BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,  # 背景 + LV + RV + MYO
        features=(32, 32, 64, 128, 256, 512),
    ).to(DEVICE)

    # 训练
    EPOCHS = 10
    n_batches = len(train_loader)
    print(f"\n开始训练 ({EPOCHS} epochs, 每 epoch {n_batches} 个 batch)...")
    print(f"  第一次 epoch 会比较慢（CPU 上约 2-5 分钟），之后会好一些")
    history = train(model, train_loader, val_loader, EPOCHS)

    # Loss 曲线
    plot_loss(history, OUTPUT_ROOT / "exp003_loss_curve.png")

    # 最终预测图
    visualize_predictions(model, val_loader, OUTPUT_ROOT / "exp003_predictions.png")

    # 保存模型
    torch.save(model.state_dict(), OUTPUT_ROOT / "exp003_baseline_unet.pth")
    print(f"\n模型已保存: {OUTPUT_ROOT / 'exp003_baseline_unet.pth'}")

    print("\n" + "=" * 60)
    print("Exp003 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
