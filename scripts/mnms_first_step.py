from pathlib import Path
import csv

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# 1. 工程路径设置
# ============================================================
# 当前脚本位置：
# M&Ms/scripts/mnms_first_step.py
#
# PROJECT_ROOT:
# M&Ms/
#
# DATASET_ROOT:
# M&Ms/dataset/
#
# OUTPUT_ROOT:
# M&Ms/outputs/
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "exp001"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 扫描 M&Ms Training/Labeled 中的病例
# ============================================================

def find_labeled_cases(dataset_root: Path):
    """
    查找 M&Ms Training/Labeled 下面的病例。

    期望结构：
    M&Ms/
    ├── dataset/
    │   └── Training/
    │       └── Labeled/
    │           └── <case_id>/
    │               ├── <case_id>_sa.nii.gz
    │               └── <case_id>_sa_gt.nii.gz
    """

    labeled_dir = dataset_root / "Training" / "Labeled"

    if not labeled_dir.exists():
        raise FileNotFoundError(
            f"没有找到目录：{labeled_dir}\n"
            f"请检查你的数据是否放在：M&Ms/dataset/Training/Labeled/"
        )

    cases = []

    for case_dir in sorted(labeled_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        image_files = list(case_dir.glob("*_sa.nii.gz"))
        label_files = list(case_dir.glob("*_sa_gt.nii.gz"))

        if len(image_files) == 0 or len(label_files) == 0:
            print(f"[跳过] {case_dir.name}：没有找到 image 或 label")
            continue

        cases.append(
            {
                "case_id": case_dir.name,
                "image_path": image_files[0],
                "label_path": label_files[0],
            }
        )

    return cases


# ============================================================
# 3. 处理 3D / 4D 数据
# ============================================================

def select_annotated_frame(image_data: np.ndarray, label_data: np.ndarray):
    """
    M&Ms 的 cine MRI 可能是 4D：
    [H, W, Z, T]

    如果是 4D：
    自动选择有标注的时间帧。

    如果是 3D：
    直接返回原数据。
    """

    if image_data.ndim == 3:
        return image_data, label_data, -1

    if image_data.ndim == 4:
        valid_frames = []

        for t in range(label_data.shape[-1]):
            if np.any(label_data[..., t] > 0):
                valid_frames.append(t)

        if len(valid_frames) == 0:
            frame_idx = 0
        else:
            frame_idx = valid_frames[0]

        image_3d = image_data[..., frame_idx]
        label_3d = label_data[..., frame_idx]

        return image_3d, label_3d, frame_idx

    raise ValueError(f"暂不支持这个维度：{image_data.shape}")


def select_best_slice(label_3d: np.ndarray):
    """
    选择标签面积最大的那一层。
    这样比直接取中间层更容易看到心脏结构。
    """

    if label_3d.ndim != 3:
        raise ValueError(f"label 应该是 3D，但现在是：{label_3d.shape}")

    areas = []

    for z in range(label_3d.shape[-1]):
        area = np.sum(label_3d[..., z] > 0)
        areas.append(area)

    best_z = int(np.argmax(areas))
    return best_z


# ============================================================
# 4. 图像显示辅助函数
# ============================================================

def normalize_for_show(image_2d: np.ndarray):
    """
    将 MRI 灰度图归一化到 0~1，方便显示。
    使用 1% 和 99% 分位数裁剪，避免极端值影响显示。
    """

    image_2d = image_2d.astype(np.float32)

    low, high = np.percentile(image_2d, 1), np.percentile(image_2d, 99)
    image_2d = np.clip(image_2d, low, high)

    if high - low < 1e-8:
        return np.zeros_like(image_2d)

    return (image_2d - low) / (high - low)


# ============================================================
# 5. 可视化单个病例
# ============================================================

def visualize_case(case_info: dict, save_path: Path = None):
    """
    对单个病例生成三联图：
    MRI image | Mask | Overlay
    """

    case_id = case_info["case_id"]
    image_path = case_info["image_path"]
    label_path = case_info["label_path"]

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))

    image_data = image_nii.get_fdata()
    label_data = label_nii.get_fdata()

    image_3d, label_3d, frame_idx = select_annotated_frame(
        image_data=image_data,
        label_data=label_data,
    )

    best_slice = select_best_slice(label_3d)

    image_2d = image_3d[..., best_slice]
    label_2d = label_3d[..., best_slice]

    image_show = normalize_for_show(image_2d)

    raw_shape = image_nii.shape
    spacing = image_nii.header.get_zooms()
    label_values = sorted(np.unique(label_3d).astype(int).tolist())

    if save_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image_show, cmap="gray")
        axes[0].set_title("MRI image")
        axes[0].axis("off")

        axes[1].imshow(label_2d, cmap="viridis")
        axes[1].set_title("Mask")
        axes[1].axis("off")

        axes[2].imshow(image_show, cmap="gray")
        axes[2].imshow(label_2d, cmap="jet", alpha=0.35)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        fig.suptitle(
            f"Case: {case_id} | raw shape: {raw_shape} | "
            f"frame: {frame_idx} | slice: {best_slice}",
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()

    summary = {
        "case_id": case_id,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "raw_shape": str(raw_shape),
        "spacing": str(spacing),
        "selected_frame": frame_idx,
        "selected_slice": best_slice,
        "label_values": str(label_values),
        "figure_path": str(save_path) if save_path else "",
    }

    return summary


# ============================================================
# 6. 保存 summary.csv
# ============================================================

def save_summary_csv(summaries: list, csv_path: Path):
    """
    不使用 pandas，直接用 Python 内置 csv 保存。
    """

    if len(summaries) == 0:
        print("没有 summary 可以保存。")
        return

    fieldnames = list(summaries[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


# ============================================================
# 7. 主函数
# ============================================================

def main():
    print("=" * 80)
    print("M&Ms First Step: 读取病例并生成 MRI / Mask / Overlay")
    print("=" * 80)

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"OUTPUT_ROOT:  {OUTPUT_ROOT}")
    print()

    cases = find_labeled_cases(DATASET_ROOT)

    print(f"找到标注病例数量：{len(cases)}")

    if len(cases) == 0:
        print("没有找到病例，请检查 dataset/Training/Labeled 目录。")
        return

    summaries = []

    # 处理全部病例
    for i, case in enumerate(cases):
        case_id = case["case_id"]

        # 前 3 例保存 overlay 图，后面的只记录数据
        if i < 3:
            save_path = OUTPUT_ROOT / f"{case_id}_mri_mask_overlay.png"
        else:
            save_path = None

        summary = visualize_case(case, save_path)
        summaries.append(summary)

        print(f"[{i+1}/{len(cases)}] {case_id}: raw shape={summary['raw_shape']}, "
              f"spacing~={summary['spacing'][:20]}..., "
              f"frame={summary['selected_frame']}, slice={summary['selected_slice']}, "
              f"labels={summary['label_values']}")

    csv_path = OUTPUT_ROOT / "mnms_first_step_summary.csv"
    save_summary_csv(summaries, csv_path)

    print()
    print("=" * 80)
    print("第一步完成。")
    print(f"summary.csv 已保存到：{csv_path}")
    print(f"图片已保存到：{OUTPUT_ROOT.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()