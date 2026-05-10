"""
第二步：扫描全部 150 例 labeled 数据，合并 vendor 信息，输出 domain shift 速查报告。

前置条件：
  1. 已跑过 mnms_first_step.py
  2. CSV 文件在 dataset/ 下

输出：
  - outputs/mnms_all_summary.csv (全部 150 例的 shape/spacing/vendor 信息)
  - outputs/vendor_comparison.png  (不同 vendor 的图像对比图)
  - 终端打印统计结果
"""

from pathlib import Path
import csv

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "exp001"

CSV_PATH = DATASET_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 读取 CSV 元数据（vendor / centre / pathology）
# ============================================================

def load_vendor_info(csv_path: Path) -> dict:
    """
    读取 CSV，返回 {case_id: {vendor, centre, pathology, ...}} 字典。
    case_id = External code（例如 'A0S9V9'）
    """
    info = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["External code"]
            info[case_id] = {
                "vendor_name": row["VendorName"],
                "vendor_letter": row["Vendor"],
                "centre": row["Centre"],
                "pathology": row["Pathology"],
                "sex": row["Sex"],
                "age": row["Age"],
            }
    return info


# ============================================================
# 2. 扫描 Training/Labeled 病例
# ============================================================

def scan_labeled_cases(dataset_root: Path, vendor_info: dict):
    """
    扫描 Training/Labeled/，对每个病例读取 NIfTI 并合并 vendor 信息。
    跳过 CSV 里找不到的病例。
    """
    labeled_dir = dataset_root / "Training" / "Labeled"
    if not labeled_dir.exists():
        raise FileNotFoundError(f"找不到 {labeled_dir}")

    results = []

    for case_dir in sorted(labeled_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        case_id = case_dir.name

        # 在 CSV 中查找 vendor 信息
        if case_id not in vendor_info:
            print(f"[跳过] {case_id}：CSV 中没有该病例")
            continue

        image_path = case_dir / f"{case_id}_sa.nii.gz"
        label_path = case_dir / f"{case_id}_sa_gt.nii.gz"

        if not image_path.exists() or not label_path.exists():
            print(f"[跳过] {case_id}：缺少 image 或 label")
            continue

        # 读取 NIfTI
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        image_data = image_nii.get_fdata()
        label_data = label_nii.get_fdata()

        shape = image_nii.shape
        spacing = image_nii.header.get_zooms()

        # 检查是 3D 还是 4D
        n_dims = image_data.ndim

        # 标签类别值
        label_values = sorted(np.unique(label_data).astype(int).tolist())

        # 有标注的 frame 数
        if n_dims == 4:
            annotated_frames = []
            for t in range(label_data.shape[-1]):
                if np.any(label_data[..., t] > 0):
                    annotated_frames.append(t)
            n_annotated = len(annotated_frames)
        elif n_dims == 3:
            n_annotated = 1 if np.any(label_data > 0) else 0
        else:
            n_annotated = 0

        # 合并 vendor 信息
        info = vendor_info[case_id]
        results.append({
            "case_id": case_id,
            "vendor": info["vendor_name"],
            "vendor_letter": info["vendor_letter"],
            "centre": info["centre"],
            "pathology": info["pathology"],
            "shape_h": shape[0],
            "shape_w": shape[1],
            "shape_n_slices": shape[2] if len(shape) >= 3 else 1,
            "shape_n_frames": shape[3] if len(shape) == 4 else 1,
            "n_dims": n_dims,
            "spacing_x": float(spacing[0]),
            "spacing_y": float(spacing[1]),
            "spacing_z": float(spacing[2]) if len(spacing) >= 3 else 0,
            "n_annotated_frames": n_annotated,
            "label_values": str(label_values),
        })

    return results


# ============================================================
# 3. 统计 & 输出
# ============================================================

def print_statistics(results: list):
    """打印关键统计信息。"""
    print(f"\n总共扫描: {len(results)} 例")
    print()

    # 按 vendor 统计
    vendors = {}
    for r in results:
        v = r["vendor"]
        if v not in vendors:
            vendors[v] = []
        vendors[v].append(r)

    print("=" * 70)
    print("按 Vendor 统计")
    print("=" * 70)
    for v, cases in sorted(vendors.items(), key=lambda x: -len(x[1])):
        shapes = set((c["shape_h"], c["shape_w"]) for c in cases)
        spacings = set((round(c["spacing_x"], 3), round(c["spacing_y"], 3)) for c in cases)
        centres = set(c["centre"] for c in cases)
        pathologies = set(c["pathology"] for c in cases)
        print(f"\n  {v} ({len(cases)} 例)")
        print(f"    Centre:      {centres}")
        print(f"    图像尺寸:     {shapes}")
        print(f"    Spacing(XY): {spacings}")
        print(f"    Pathology:   {pathologies}")

    print()

    # 总体统计
    print("=" * 70)
    print("总体统计")
    print("=" * 70)

    all_shapes = set((r["shape_h"], r["shape_w"]) for r in results)
    all_spacings = set((round(r["spacing_x"], 3), round(r["spacing_y"], 3)) for r in results)
    print(f"  图像种类: {len(all_shapes)} 种不同尺寸")
    print(f"  Spacing 种类: {len(all_spacings)} 种不同 spacing")

    n_4d = sum(1 for r in results if r["n_dims"] == 4)
    n_3d = sum(1 for r in results if r["n_dims"] == 3)
    print(f"  4D 数据: {n_4d} 例")
    print(f"  3D 数据: {n_3d} 例")

    n_one_frame = sum(1 for r in results if r["n_annotated_frames"] == 1)
    n_two_frame = sum(1 for r in results if r["n_annotated_frames"] >= 2)
    print(f"  只有 1 帧标注: {n_one_frame} 例")
    print(f"  有 2 帧+ 标注: {n_two_frame} 例")


# ============================================================
# 4. Vendor 对比图
# ============================================================

def plot_vendor_comparison(results: list, save_path: Path):
    """
    每个 vendor 选 1 个代表性病例，展示 MRI + mask overlay。
    用于直观对比不同 vendor 的图像外观差异。
    """
    # 每个 vendor 选第一个有数据的病例
    seen = {}
    for r in results:
        v = r["vendor"]
        if v not in seen:
            seen[v] = r

    if len(seen) < 2:
        print("病例太少，无法生成对比图")
        return

    vendors_ordered = sorted(seen.keys())
    n_vendors = len(vendors_ordered)

    fig, axes = plt.subplots(2, n_vendors, figsize=(4 * n_vendors, 8))

    for i, v in enumerate(vendors_ordered):
        case = seen[v]
        case_id = case["case_id"]
        labeled_dir = DATASET_ROOT / "Training" / "Labeled" / case_id
        image_path = labeled_dir / f"{case_id}_sa.nii.gz"
        label_path = labeled_dir / f"{case_id}_sa_gt.nii.gz"

        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        image_data = image_nii.get_fdata()
        label_data = label_nii.get_fdata()

        # 选有标注的 frame
        if image_data.ndim == 4:
            for t in range(label_data.shape[-1]):
                if np.any(label_data[..., t] > 0):
                    image_3d = image_data[..., t]
                    label_3d = label_data[..., t]
                    break
            else:
                image_3d = image_data[..., 0]
                label_3d = label_data[..., 0]
        else:
            image_3d = image_data
            label_3d = label_data

        # 选标签面积最大的 slice
        areas = [np.sum(label_3d[..., z] > 0) for z in range(label_3d.shape[-1])]
        best_z = int(np.argmax(areas))

        image_2d = image_3d[..., best_z].astype(np.float32)
        label_2d = label_3d[..., best_z]

        # 归一化显示
        low, high = np.percentile(image_2d, 1), np.percentile(image_2d, 99)
        image_2d = np.clip(image_2d, low, high)
        image_show = (image_2d - low) / (high - low) if high > low else np.zeros_like(image_2d)

        # MRI
        axes[0, i].imshow(image_show, cmap="gray")
        axes[0, i].set_title(f"{v}\nCentre {case['centre']}\n{image_2d.shape[0]}x{image_2d.shape[1]}")
        axes[0, i].axis("off")

        # Overlay
        axes[1, i].imshow(image_show, cmap="gray")
        axes[1, i].imshow(label_2d, cmap="jet", alpha=0.35)
        axes[1, i].set_title(f"Overlay\n{case['pathology']}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Vendor 对比图已保存: {save_path}")


# ============================================================
# 5. 保存 summary CSV
# ============================================================

def save_summary(results: list, csv_path: Path):
    fieldnames = [
        "case_id", "vendor", "vendor_letter", "centre", "pathology",
        "shape_h", "shape_w", "shape_n_slices", "shape_n_frames",
        "n_dims", "spacing_x", "spacing_y", "spacing_z",
        "n_annotated_frames", "label_values",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Summary 已保存: {csv_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("读取 CSV 元数据...")
    vendor_info = load_vendor_info(CSV_PATH)
    print(f"CSV 中共 {len(vendor_info)} 例")

    print("扫描 Training/Labeled...")
    results = scan_labeled_cases(DATASET_ROOT, vendor_info)

    print_statistics(results)

    csv_path = OUTPUT_ROOT / "mnms_all_summary.csv"
    save_summary(results, csv_path)

    print("\n生成 Vendor 对比图...")
    plot_vendor_comparison(results, OUTPUT_ROOT / "vendor_comparison.png")

    print("\n全部完成！")


if __name__ == "__main__":
    main()
