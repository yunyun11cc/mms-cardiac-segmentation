"""
Exp004: Vendor Shift 实验 — Leave-One-Vendor-Out

目标：量化不同厂商之间的分割性能差距

实验设计：
  Split 1: 训练 Siemens → 测试 Philips
  Split 2: 训练 Philips → 测试 Siemens
  Split 3: 训练 Siemens+Philips → 测试 GE+Canon（unseen vendor）

输出：
  - exp004_dice_table.txt：每个 Split 的 Dice 结果表
  - exp004_dice_barplot.png：柱状对比图
  - exp004_split3_predictions.png：Unseen vendor 预测图
"""

from pathlib import Path
import csv

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from monai.transforms import (
    Compose, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, CropForegroundd, SpatialPadd,
)
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.data import DataLoader, Dataset
from monai.metrics import DiceMetric

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "exp004"
CSV_PATH = DATASET_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}")


# ============================================================
# 1. 加载 CSV 获取 vendor 信息
# ============================================================

def load_vendor_map():
    """返回 {case_id: vendor_name}"""
    vm = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            vm[r["External code"]] = r["VendorName"]
    return vm


# ============================================================
# 2. 选 ED frame（复用）
# ============================================================

def select_ed_frame(image_data, label_data):
    if image_data.ndim == 4:
        for t in range(label_data.shape[-1]):
            if np.any(label_data[..., t] > 0):
                return image_data[..., t], label_data[..., t], t
        return image_data[..., 0], label_data[..., 0], 0
    return image_data, label_data, -1


# ============================================================
# 3. 预处理 pipeline
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
# 4. 加载指定 vendor 的所有 slice
# ============================================================

def load_vendor_slices(case_ids, pipeline):
    """加载一批病例，返回 image slices 和 label slices"""
    images, labels = [], []
    for case_id in case_ids:
        for split in ["Training/Labeled", "Validation", "Testing"]:
            img_path = DATASET_ROOT / split / case_id / f"{case_id}_sa.nii.gz"
            lbl_path = DATASET_ROOT / split / case_id / f"{case_id}_sa_gt.nii.gz"
            if img_path.exists():
                break
        else:
            continue

        try:
            img_nii = nib.load(str(img_path))
            lbl_nii = nib.load(str(lbl_path))
            img_4d = img_nii.get_fdata()
            lbl_4d = lbl_nii.get_fdata()
        except Exception:
            continue

        img_3d, lbl_3d, _ = select_ed_frame(img_4d, lbl_4d)

        try:
            out = pipeline({"image": img_3d, "label": lbl_3d})
            vol_img = out["image"].numpy()
            vol_lbl = out["label"].numpy()
        except Exception:
            continue

        n_before = len(images)
        for z in range(vol_lbl.shape[-1]):
            lbl_slice = vol_lbl[0, :, :, z]
            if np.sum(lbl_slice > 0) > 20:
                images.append(vol_img[:, :, :, z].copy())      # [1, 256, 256]
                labels.append(lbl_slice[None, :, :].copy())   # [1, 256, 256]
        n_this = len(images) - n_before

        print(f"    {case_id}: {vol_img.shape} → {n_this} slices")

    print(f"  → {len(images)} slices")
    return images, labels


def get_cases_by_vendor(vendor_map, vendors):
    """返回属于指定 vendor 的 case_id 列表"""
    return [cid for cid, v in vendor_map.items() if v in vendors]


# ============================================================
# 5. 训练 + 评估
# ============================================================

def train_and_eval(train_ids, test_ids, split_name, vendor_map, pipeline):
    print(f"\n{'='*60}")
    print(f"Split: {split_name}")
    print(f"{'='*60}")

    print("加载训练数据...")
    train_imgs, train_lbls = load_vendor_slices(train_ids, pipeline)

    print("加载测试数据...")
    test_imgs, test_lbls = load_vendor_slices(test_ids, pipeline)

    if len(train_imgs) == 0 or len(test_imgs) == 0:
        print("  [跳过] 数据不足")
        return None

    train_ds = Dataset(data=[{"image": i, "label": l} for i, l in zip(train_imgs, train_lbls)])
    test_ds = Dataset(data=[{"image": i, "label": l} for i, l in zip(test_imgs, test_lbls)])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    model = BasicUNet(spatial_dims=2, in_channels=1, out_channels=4,
                      features=(32, 32, 64, 128, 256, 512)).to(DEVICE)

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None

    for epoch in tqdm(range(10), desc=f"  {split_name}"):
        model.train()
        for batch in tqdm(train_loader, desc="    Train", leave=False):
            x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
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

    # 评估
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean_per_channel")
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            y_pred = torch.softmax(model(x), dim=1)
            # y: [B, 1, H, W] → squeeze → [B, H, W] → one_hot → [B, H, W, 4] → permute → [B, 4, H, W]
            y_onehot = torch.nn.functional.one_hot(
                y.squeeze(1).long(), num_classes=4
            ).permute(0, 3, 1, 2)
            dice_metric(y_pred, y_onehot)

    dice_per_class = dice_metric.aggregate().cpu().numpy()  # [LV, RV, MYO] (bg excluded)
    dice_avg = np.mean(dice_per_class)
    print(f"  Dice: LV={dice_per_class[0]:.4f}, RV={dice_per_class[1]:.4f}, MYO={dice_per_class[2]:.4f}")
    print(f"  Avg Dice: {dice_avg:.4f}")

    return {"name": split_name, "dice": dice_per_class, "avg": dice_avg}


# ============================================================
# 6. 结果可视化
# ============================================================

def plot_results(results, save_path):
    """柱状图对比各 Split 的 Dice"""
    names = [r["name"] for r in results]
    lv = [r["dice"][0] for r in results]
    rv = [r["dice"][1] for r in results]
    myo = [r["dice"][2] for r in results]

    x = np.arange(len(names))
    w = 0.22

    # 自动计算 y 轴范围，留 20% 顶部空间放数值标注
    all_vals = lv + rv + myo
    y_max = max(all_vals) * 1.5 if max(all_vals) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - w, lv, w, label="LV", color="#2E86AB")
    bars2 = ax.bar(x, rv, w, label="RV", color="#A23B72")
    bars3 = ax.bar(x + w, myo, w, label="MYO", color="#F18F01")

    # 柱顶标数值
    def add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + y_max*0.02,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    # 缩短 x 轴标签（太长会重叠）
    short_names = [n.replace("Train ", "").replace("Test ", "").replace("Siemens+Philips", "Siem+Phi")
                   .replace("Mixed (Siemens+Philips)", "Mixed\n(Siem+Phi)") for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)

    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title("Leave-One-Vendor-Out — Per-Class Dice", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"柱状图保存: {save_path}")


def save_results_table(results, save_path):
    """保存 Dice 结果表"""
    lines = ["Leave-One-Vendor-Out Results", "=" * 50, ""]
    lines.append(f"{'Split':<25} {'LV':<8} {'RV':<8} {'MYO':<8} {'Avg':<8}")
    lines.append("-" * 50)
    for r in results:
        lv, rv, myo = r["dice"][0], r["dice"][1], r["dice"][2]
        lines.append(f"{r['name']:<25} {lv:<8.4f} {rv:<8.4f} {myo:<8.4f} {r['avg']:<8.4f}")
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"结果表保存: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Exp004: Leave-One-Vendor-Out 实验")
    print("=" * 60)

    vendor_map = load_vendor_map()
    pipeline = build_pipeline()

    # 获取各 vendor 的病例
    siemens = get_cases_by_vendor(vendor_map, ["Siemens"])
    philips = get_cases_by_vendor(vendor_map, ["Philips"])
    ge_canon = get_cases_by_vendor(vendor_map, ["GE", "Canon"])
    print(f"Siemens: {len(siemens)} 例, Philips: {len(philips)} 例, GE+Canon: {len(ge_canon)} 例")

    splits = [
        ("Train Siemens → Test Philips", siemens, philips),
        ("Train Philips → Test Siemens", philips, siemens),
        ("Train Siemens+Philips → Test GE+Canon", siemens + philips, ge_canon),
    ]

    results = []
    for name, train_ids, test_ids in splits:
        r = train_and_eval(train_ids, test_ids, name, vendor_map, pipeline)
        if r:
            results.append(r)

    if results:
        plot_results(results, OUTPUT_ROOT / "exp004_dice_barplot.png")
        save_results_table(results, OUTPUT_ROOT / "exp004_dice_table.txt")

    # 同时也训练一个混合域模型做参照
    print("\n" + "=" * 60)
    print("训练混合域模型 (Siemens+Philips) 作为参照...")
    all_train = siemens + philips
    # 用 80% 训练，20% 验证
    split_idx = int(len(all_train) * 0.8)
    ref = train_and_eval(all_train[:split_idx], all_train[split_idx:],
                         "Mixed (Siemens+Philips)", vendor_map, pipeline)
    if ref:
        results.append(ref)
        plot_results(results, OUTPUT_ROOT / "exp004_dice_barplot.png")
        save_results_table(results, OUTPUT_ROOT / "exp004_dice_table.txt")

    print("\n" + "=" * 60)
    print("Exp004 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
