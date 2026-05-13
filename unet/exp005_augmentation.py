"""
Exp005: 数据增强缓解 Domain Shift

目标：用强度增强提升 unseen vendor 的 Dice

实验设计：
  在 Train S+P → Test GE+Canon 上对比：
  1. Split3 baseline（无增强）
  2. Split3 强度增强（随机亮度/对比度/噪声）

输出：
  - outputs/unet/exp005/exp005_intensity_dice_table.txt
  - outputs/unet/exp005/exp005_intensity_dice_barplot.png
"""

from pathlib import Path
import csv

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose, EnsureChannelFirstd, Spacingd,
    NormalizeIntensityd, CropForegroundd, SpatialPadd,
)
from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.data import DataLoader, Dataset, pad_list_data_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "unet" / "exp005"
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
print(f"设备: {DEVICE}")


# ============================================================
# 1. 工具函数（复用 Exp004）
# ============================================================

def load_vendor_map():
    vm = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            vm[r["External code"]] = r["VendorName"]
    return vm


def get_cases_by_vendor(vendor_map, vendors):
    return [cid for cid, v in vendor_map.items() if v in vendors]


def select_ed_frame(image_data, label_data):
    if image_data.ndim == 4:
        for t in range(label_data.shape[-1]):
            if np.any(label_data[..., t] > 0):
                return image_data[..., t], label_data[..., t], t
        return image_data[..., 0], label_data[..., 0], 0
    return image_data, label_data, -1


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
# 2. 强度增强（核心新功能）
# ============================================================

def augment_intensity(image_2d, rng, prob=0.8, scale_range=(0.8, 1.2),
                      shift_range=(-0.15, 0.15), noise_std=0.05):
    """对单张 2D 图像做随机亮度/对比度/噪声扰动（只在非零区域）"""
    if rng.random() >= prob:
        return image_2d.copy()

    out = image_2d.copy().astype(np.float32)
    mask = out != 0
    if not mask.any():
        return out

    scale = rng.uniform(*scale_range)
    shift = rng.uniform(*shift_range)
    out[mask] = out[mask] * scale + shift

    if noise_std > 0:
        noise = rng.normal(0, rng.uniform(0, noise_std), size=mask.sum())
        out[mask] = out[mask] + noise.astype(np.float32)

    return out


class IntensityAugmentd:
    """MONAI 字典 transform：对 'image' 键做强度增强"""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        if torch.is_tensor(img):
            img_np = img.numpy()
            aug = augment_intensity(img_np, self.rng)
            d["image"] = torch.as_tensor(aug, dtype=img.dtype, device=img.device)
        else:
            d["image"] = augment_intensity(img, self.rng)
        return d


# ============================================================
# 3. 加载数据
# ============================================================

def load_vendor_slices(case_ids, pipeline):
    images, labels = [], []
    for case_id in case_ids:
        # 找到文件所在目录
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
                labels.append(lbl_slice[None, :, :].copy())

        n_slices = sum(1 for z in range(vol_lbl.shape[-1]) if np.sum(vol_lbl[0, :, :, z] > 0) > 20)
        print(f"    {case_id}: {vol_img.shape} -> {n_slices} slices")

    print(f"  Total: {len(images)} slices")
    return images, labels


# ============================================================
# 4. 训练 + 评估
# ============================================================

def train_and_eval(train_ids, test_ids, name, pipeline, train_transform=None):
    import time

    print(f"\n{'='*60}")
    print(f"Variant: {name}")
    print(f"{'='*60}")

    print("加载训练数据...")
    train_imgs, train_lbls = load_vendor_slices(train_ids, pipeline)
    print("加载测试数据...")
    test_imgs, test_lbls = load_vendor_slices(test_ids, pipeline)

    if len(train_imgs) == 0 or len(test_imgs) == 0:
        print("  [跳过] 数据不足")
        return None

    train_ds = Dataset(
        data=[{"image": i, "label": l} for i, l in zip(train_imgs, train_lbls)],
        transform=train_transform,
    )
    test_ds = Dataset(
        data=[{"image": i, "label": l} for i, l in zip(test_imgs, test_lbls)],
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0,
                              collate_fn=pad_list_data_collate)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0,
                             collate_fn=pad_list_data_collate)

    model = BasicUNet(spatial_dims=2, in_channels=1, out_channels=4,
                      features=(32, 32, 64, 128, 256, 512)).to(DEVICE)

    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    # --- 训练 ---
    epochs = 10
    for epoch in tqdm(range(epochs), desc="Epoch"):
        epoch_start = time.time()
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc="  Train", leave=False):
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

            train_loss += loss.item()

        et = time.time() - epoch_start
        tqdm.write(f"  Epoch {epoch+1}/{epochs} | loss={train_loss/len(train_loader):.4f} | {et:.0f}s")

    # --- 评估 ---
    model.eval()
    dice_sums = np.zeros(3, dtype=np.float64)
    dice_count = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            y_pred = torch.argmax(model(x), dim=1).cpu().numpy()
            y_true = y[:, 0].long().cpu().numpy()

            for c in range(3):  # 0=LV, 1=RV, 2=MYO
                pred_c = y_pred == (c + 1)
                true_c = y_true == (c + 1)
                inter = (pred_c & true_c).sum()
                denom = pred_c.sum() + true_c.sum()
                dice_sums[c] += (2 * inter + 1e-6) / (denom + 1e-6)
            dice_count += 1

    dice_per_class = dice_sums / dice_count
    dice_avg = np.mean(dice_per_class)
    print(f"  Dice: LV={dice_per_class[0]:.4f}, RV={dice_per_class[1]:.4f}, MYO={dice_per_class[2]:.4f}")
    print(f"  Avg Dice: {dice_avg:.4f}")

    return {"name": name, "dice": dice_per_class, "avg": dice_avg}


# ============================================================
# 5. 结果可视化
# ============================================================

def plot_results(results, save_path):
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
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.80, color="gray", linestyle="--", alpha=0.5, label="Exp004 Mixed Ref")
    ax.legend()
    ax.set_title("Exp005: Intensity Augmentation vs Baseline")

    # 柱顶标数值
    for i in range(len(names)):
        for bar, vals in [(x[i] - w, lv[i]), (x[i], rv[i]), (x[i] + w, myo[i])]:
            ax.text(bar, vals + 0.02, f"{vals:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"柱状图保存: {save_path}")


def save_results_table(results, save_path):
    lines = ["Exp005 Intensity Augmentation Results", "=" * 58, ""]
    lines.append(f"{'Variant':<32} {'LV':<8} {'RV':<8} {'MYO':<8} {'Avg':<8}")
    lines.append("-" * 58)
    for r in results:
        d = r["dice"]
        lines.append(f"{r['name']:<32} {d[0]:<8.4f} {d[1]:<8.4f} {d[2]:<8.4f} {r['avg']:<8.4f}")
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"结果表保存: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Exp005: 强度增强缓解 Domain Shift")
    print("=" * 60)

    vendor_map = load_vendor_map()
    siemens = get_cases_by_vendor(vendor_map, ["Siemens"])
    philips = get_cases_by_vendor(vendor_map, ["Philips"])
    ge_canon = get_cases_by_vendor(vendor_map, ["GE", "Canon"])
    print(f"Siemens: {len(siemens)}, Philips: {len(philips)}, GE+Canon: {len(ge_canon)}")

    train_ids = siemens + philips
    test_ids = ge_canon
    pipeline = build_pipeline()
    results = []

    # 1. Baseline（无增强）
    r = train_and_eval(train_ids, test_ids, "Split3 baseline", pipeline)
    if r:
        results.append(r)

    # 2. 强度增强
    aug = IntensityAugmentd(seed=1042)
    r = train_and_eval(train_ids, test_ids, "Split3 + intensity aug", pipeline,
                       train_transform=aug)
    if r:
        results.append(r)

    if results:
        save_results_table(results, OUTPUT_ROOT / "exp005_intensity_dice_table.txt")
        plot_results(results, OUTPUT_ROOT / "exp005_intensity_dice_barplot.png")

    print("\n" + "=" * 60)
    print("Exp005 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
