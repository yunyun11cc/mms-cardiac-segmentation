"""
Exp002: 预处理 Pipeline

目标：把形状不一致的 NIfTI 转为统一 [1, 256, 256] 的张量

流程：
  Load NIfTI → 选 ED frame → Spacing(resample to 1.5mm)
  → NormalizeIntensity(z-score) → CropForeground(心脏区域)
  → SpatialPad(256x256) → [1, 256, 256]

验证方式：
  1. 读一个病例
  2. 处理前后对比图
  3. 打印 shape 变化
"""

from pathlib import Path
import csv

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    Lambdad,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "exp002"
CSV_PATH = DATASET_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 读取 CSV，找病例
# ============================================================

def get_case_info(limit_vendor=None):
    """从 CSV 获取病例 vendor 信息，返回 {case_id: vendor, centre} 字典"""
    info = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            info[r["External code"]] = {
                "vendor": r["VendorName"],
                "centre": r["Centre"],
            }
    return info


# ============================================================
# 2. 选 ED frame（复用之前逻辑）
# ============================================================

def select_annotated_frame(image_data, label_data):
    """
    从 4D 中选有标注的第一帧（通常是 ED 帧）。
    3D 则直接返回。
    """
    if image_data.ndim == 4:
        for t in range(label_data.shape[-1]):
            if np.any(label_data[..., t] > 0):
                return image_data[..., t], label_data[..., t], t
        return image_data[..., 0], label_data[..., 0], 0
    return image_data, label_data, -1


def select_best_slice(label_3d):
    """选标签面积最大的 z-slice（用于可视化，训练时用全部 slice）。"""
    areas = [np.sum(label_3d[..., z] > 0) for z in range(label_3d.shape[-1])]
    return int(np.argmax(areas))


# ============================================================
# 3. MONAI 预处理 Pipeline
# ============================================================

def build_pipeline():
    """
    构建 MONAI 预处理链。

    输入 dict: {"image": 3D array [H, W, Z], "label": 3D array [H, W, Z]}
    输出 dict: {"image": [1, H', W'], "label": [1, H', W']}
    """
    return Compose([
        # 添加 channel 维： [H, W, Z] → [1, H, W, Z]
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),

        # Resample 到 1.5mm 各向同性（bilinear for image, nearest for label）
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),

        # Z-score 归一化（只在非零体素上算）
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # 裁剪到心脏区域（基于 label）
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            select_fn=lambda x: x > 0,
        ),

        # Pad/Crop 到统一 3D 尺寸 [1, 256, 256, Z']
        # 注：Z 维度固定取中间切片，用于 2D 训练
        SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, -1)),
    ])


# ============================================================
# 4. 可视化对比
# ============================================================

def visualize_preprocessing(case_id, image_before, label_before, image_after, label_after, z_before, z_after, save_path):
    """处理前 vs 处理后，slice 对比。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    def normalize(img):
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        return np.clip(img, lo, hi)

    # 处理前
    img_slice = image_before[..., z_before]
    img_norm = normalize(img_slice)
    axes[0, 0].imshow(img_norm, cmap="gray")
    axes[0, 0].set_title(f"Before: MRI\nshape={image_before.shape[:2]}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(label_before[..., z_before], cmap="viridis")
    axes[0, 1].set_title("Label")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_norm, cmap="gray")
    axes[0, 2].imshow(label_before[..., z_before], cmap="jet", alpha=0.3)
    axes[0, 2].set_title("Overlay")
    axes[0, 2].axis("off")

    # 处理后
    img_slice_a = image_after[0, :, :, z_after]
    img_slice_a = normalize(img_slice_a)
    axes[1, 0].imshow(img_slice_a, cmap="gray")
    axes[1, 0].set_title(f"After: MRI\nshape={img_slice_a.shape[:2]}")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(label_after[0, :, :, z_after], cmap="viridis")
    axes[1, 1].set_title("Label")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img_slice_a, cmap="gray")
    axes[1, 2].imshow(label_after[0, :, :, z_after], cmap="jet", alpha=0.3)
    axes[1, 2].set_title("Overlay")
    axes[1, 2].axis("off")

    fig.suptitle(f"Preprocessing: {case_id}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"对比图保存: {save_path}")


# ============================================================
# 5. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("Exp002: 预处理 Pipeline 验证")
    print("=" * 60)

    # 构建 pipeline
    pipeline = build_pipeline()

    # 选两个代表性的病例：Siemens（小图）和 Philips（大图）
    test_cases = [
        ("A0S9V9", "Siemens"),  # 216x256
        ("A1D0Q7", "Philips"),  # 320x320
    ]

    for case_id, vendor in test_cases:
        print(f"\n--- 处理: {case_id} ({vendor}) ---")

        image_path = DATASET_ROOT / "Training" / "Labeled" / case_id / f"{case_id}_sa.nii.gz"
        label_path = DATASET_ROOT / "Training" / "Labeled" / case_id / f"{case_id}_sa_gt.nii.gz"

        if not image_path.exists():
            print(f"  [跳过] 路径不存在: {image_path}")
            continue

        # 加载 NIfTI
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))
        image_data = image_nii.get_fdata()
        label_data = label_nii.get_fdata()

        # 打印原始信息
        print(f"  原始 shape: {image_data.shape}")
        print(f"  原始 spacing: {image_nii.header.get_zooms()}")

        # 选 ED frame
        image_3d, label_3d, frame_idx = select_annotated_frame(image_data, label_data)
        print(f"  ED frame: {frame_idx}, frame shape: {image_3d.shape}")

        # 选最佳 slice（用于可视化）
        best_z = select_best_slice(label_3d)

        # 预处理前备份（用于对比）
        image_before = image_3d.copy()
        label_before = label_3d.copy()

        # 执行 MONAI pipeline
        # 输入: {"image": [H,W,Z], "label": [H,W,Z]}
        # MONAI 内部会自动处理成 tensor
        input_dict = {"image": image_3d, "label": label_3d}
        output_dict = pipeline(input_dict)

        image_after = output_dict["image"].numpy()  # [1, H, W, Z']
        label_after = output_dict["label"].numpy()  # [1, H, W, Z']

        print(f"  处理后 shape: {image_after.shape}")
        print(f"  处理后 spacing: 1.5mm isotropic")
        print(f"  处理后 intensity 范围: {image_after.min():.3f} ~ {image_after.max():.3f}")
        print(f"  Label 类别: {np.unique(label_after).tolist()}")

        # 可视化
        z_after = select_best_slice(label_after[0])  # [1, H, W, Z] 中取 Z
        save_path = OUTPUT_ROOT / f"{case_id}_preprocess_compare.png"
        visualize_preprocessing(case_id, image_before, label_before, image_after, label_after,
                                z_before=best_z, z_after=z_after, save_path=save_path)

        # 验证输出
        assert image_after.shape[1:3] == (256, 256), f"尺寸不对: {image_after.shape}"
        assert label_after.shape[1:3] == (256, 256), f"尺寸不对: {label_after.shape}"
        assert np.all(np.unique(label_after) == np.unique(label_before)), "Label 类别变了!"
        print(f"  ✅ 验证通过: 输出 (1, 256, 256, {image_after.shape[-1]})")

    print("\n" + "=" * 60)
    print("Exp002 验证完成！")
    print("下一步: 将 pipeline 集成到 DataLoader 中训练 (Exp003)")
    print("=" * 60)


if __name__ == "__main__":
    main()
