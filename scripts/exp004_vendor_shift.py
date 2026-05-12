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
import warnings
warnings.filterwarnings("ignore")

import nibabel as nib
import numpy as np
import torch

from monai.transforms import (
    Compose, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, CropForegroundd, SpatialPadd,
)
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.data import DataLoader, Dataset, pad_list_data_collate

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

        for z in range(vol_lbl.shape[-1]):
            lbl_slice = vol_lbl[0, :, :, z]
            if np.sum(lbl_slice > 0) > 20:
                images.append(vol_img[:, :, :, z].copy())
                labels.append(lbl_slice[None, :, :].copy())  # [1, 256, 256]

        n_slices = sum(1 for z in range(vol_lbl.shape[-1]) if np.sum(vol_lbl[0, :, :, z] > 0) > 20)
        print(f"    {case_id}: {vol_img.shape} → {n_slices} slices")

    print(f"  → {len(images)} slices")
    return images, labels


def get_cases_by_vendor(vendor_map, vendors):
    """返回属于指定 vendor 的 case_id 列表"""
    return [cid for cid, v in vendor_map.items() if v in vendors]


# ============================================================
# 5. 训练 + 评估
# ============================================================

def train_and_eval(train_ids, test_ids, split_name, vendor_map, pipeline):
    import time
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

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
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0,
                              collate_fn=pad_list_data_collate)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0,
                             collate_fn=pad_list_data_collate)

    model = BasicUNet(spatial_dims=2, in_channels=1, out_channels=4,
                      features=(32, 32, 64, 128, 256, 512)).to(DEVICE)

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练（带进度条）
    epochs = 6
    epoch_iter = tqdm(range(epochs), desc="Epoch") if has_tqdm else range(epochs)

    for epoch in epoch_iter:
        epoch_start = time.time()
        model.train()
        train_loss = 0

        batch_iter = tqdm(train_loader, desc="  Train", leave=False) if has_tqdm else train_loader

        for batch in batch_iter:
            x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        if has_tqdm:
            epoch_iter.set_postfix({"loss": f"{avg_loss:.4f}", "time": f"{epoch_time:.0f}s"})
        else:
            print(f"  Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | {epoch_time:.0f}s")

    # 评估（手写 Dice，不依赖 MONAI DiceMetric 版本兼容）
    model.eval()
    dice_sums = torch.zeros(3, device=DEVICE)  # LV, RV, MYO
    dice_counts = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            logits = model(x)                           # [B, 4, H, W]
            y_pred = torch.argmax(logits, dim=1)         # [B, H, W]
            y_true = y[:, 0].long()                      # [B, H, W]

            for c in range(1, 4):  # class 1=LV, 2=RV, 3=MYO
                pred_c = (y_pred == c)
                true_c = (y_true == c)
                inter = (pred_c & true_c).sum().float()
                dice = (2 * inter + 1e-6) / (pred_c.sum() + true_c.sum() + 1e-6)
                dice_sums[c - 1] += dice
            dice_counts += 1

    dice_per_class = dice_sums.cpu().numpy() / dice_counts
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
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, lv, w, label="LV")
    ax.bar(x, rv, w, label="RV")
    ax.bar(x + w, myo, w, label="MYO")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_title("Leave-One-Vendor-Out Dice Comparison")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
