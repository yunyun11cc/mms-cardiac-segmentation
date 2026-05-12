"""
Exp005: Intensity augmentation for unseen-vendor generalization.

Experiment:
  Train Siemens+Philips -> Test GE+Canon

Variants:
  1. Split3 baseline without augmentation
  2. Split3 with intensity augmentation on training slices only

Outputs:
  - outputs/exp005/exp005_intensity_dice_table.txt
  - outputs/exp005/exp005_intensity_dice_barplot.png
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.losses import DiceLoss
from monai.networks.nets import BasicUNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Spacingd,
    SpatialPadd,
)
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "exp005"
CSV_PATH = DATASET_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"


@dataclass(frozen=True)
class IntensityAugmentConfig:
    """Intensity-only augmentation ranges for normalized MRI slices."""

    prob: float = 0.8
    scale_range: tuple[float, float] = (0.8, 1.2)
    shift_range: tuple[float, float] = (-0.15, 0.15)
    noise_std_range: tuple[float, float] = (0.0, 0.05)


def augment_image_intensity(
    image: np.ndarray,
    rng: np.random.Generator,
    config: IntensityAugmentConfig,
) -> np.ndarray:
    """Apply scale, shift, and Gaussian noise to nonzero image pixels."""

    output = np.asarray(image, dtype=np.float32).copy()
    if rng.random() >= config.prob:
        return output

    foreground = output != 0
    if not np.any(foreground):
        return output

    scale = rng.uniform(*config.scale_range)
    shift = rng.uniform(*config.shift_range)
    noise_std = rng.uniform(*config.noise_std_range)

    output[foreground] = output[foreground] * scale + shift
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=int(foreground.sum())).astype(np.float32)
        output[foreground] = output[foreground] + noise

    return output.astype(np.float32, copy=False)


class IntensityAugmentd:
    """Dictionary transform that augments only the image field."""

    def __init__(
        self,
        config: IntensityAugmentConfig | None = None,
        seed: int | None = None,
        image_key: str = "image",
    ) -> None:
        self.config = config or IntensityAugmentConfig()
        self.rng = np.random.default_rng(seed)
        self.image_key = image_key

    def __call__(self, data: dict) -> dict:
        item = dict(data)
        image = item[self.image_key]

        if torch.is_tensor(image):
            device = image.device
            dtype = image.dtype
            image_np = image.detach().cpu().numpy()
            augmented = augment_image_intensity(image_np, self.rng, self.config)
            item[self.image_key] = torch.as_tensor(augmented, dtype=dtype, device=device)
        else:
            item[self.image_key] = augment_image_intensity(image, self.rng, self.config)

        return item


def compute_dice_per_class(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    eps: float = 1e-6,
) -> np.ndarray:
    """Return Dice for classes 1..num_classes-1."""

    pred_np = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else np.asarray(y_pred)
    true_np = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.asarray(y_true)

    dice_values = []
    for class_id in range(1, num_classes):
        pred_mask = pred_np == class_id
        true_mask = true_np == class_id
        denom = pred_mask.sum() + true_mask.sum()
        if denom == 0:
            dice_values.append(1.0)
            continue
        intersection = np.logical_and(pred_mask, true_mask).sum()
        dice_values.append((2.0 * intersection) / (denom + eps))

    return np.asarray(dice_values, dtype=np.float32)


def get_device() -> tuple[str, bool]:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vendor_map() -> dict[str, str]:
    """Return {case_id: vendor_name} from the M&Ms metadata CSV."""

    vendor_map = {}
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vendor_map[row["External code"]] = row["VendorName"]
    return vendor_map


def get_cases_by_vendor(vendor_map: dict[str, str], vendors: list[str]) -> list[str]:
    return [case_id for case_id, vendor in vendor_map.items() if vendor in vendors]


def select_ed_frame(image_data: np.ndarray, label_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if image_data.ndim == 4:
        for frame_idx in range(label_data.shape[-1]):
            if np.any(label_data[..., frame_idx] > 0):
                return image_data[..., frame_idx], label_data[..., frame_idx], frame_idx
        return image_data[..., 0], label_data[..., 0], 0
    return image_data, label_data, -1


def build_pipeline() -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                select_fn=lambda x: x > 0,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, -1)),
        ]
    )


def find_case_paths(case_id: str) -> tuple[Path, Path] | None:
    for split in ("Training/Labeled", "Validation", "Testing"):
        image_path = DATASET_ROOT / split / case_id / f"{case_id}_sa.nii.gz"
        label_path = DATASET_ROOT / split / case_id / f"{case_id}_sa_gt.nii.gz"
        if image_path.exists() and label_path.exists():
            return image_path, label_path
    return None


def load_vendor_slices(
    case_ids: list[str],
    pipeline: Compose,
    min_label_pixels: int = 20,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load ED-frame labeled 2D slices for the requested cases."""

    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for case_id in tqdm(case_ids, desc="  Preprocess", unit="case"):
        paths = find_case_paths(case_id)
        if paths is None:
            continue

        image_path, label_path = paths
        try:
            image_nii = nib.load(str(image_path))
            label_nii = nib.load(str(label_path))
            image_4d = image_nii.get_fdata()
            label_4d = label_nii.get_fdata()
        except Exception as exc:
            print(f"    [skip] {case_id}: failed to load NIfTI ({exc})")
            continue

        image_3d, label_3d, _ = select_ed_frame(image_4d, label_4d)

        try:
            output = pipeline({"image": image_3d, "label": label_3d})
            volume_image = output["image"].numpy()
            volume_label = output["label"].numpy()
        except Exception as exc:
            print(f"    [skip] {case_id}: preprocessing failed ({exc})")
            continue

        slice_count = 0
        for z_idx in range(volume_label.shape[-1]):
            label_slice = volume_label[0, :, :, z_idx]
            if np.sum(label_slice > 0) > min_label_pixels:
                images.append(volume_image[:, :, :, z_idx].astype(np.float32, copy=True))
                labels.append(label_slice[None, :, :].astype(np.int64, copy=True))
                slice_count += 1

        print(f"    {case_id}: {volume_image.shape} -> {slice_count} slices")

    print(f"  Total: {len(images)} slices")
    return images, labels


def build_model(device: str) -> BasicUNet:
    return BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        features=(32, 32, 64, 128, 256, 512),
    ).to(device)


def train_model(
    model: BasicUNet,
    train_loader: DataLoader,
    epochs: int,
    device: str,
    use_amp: bool,
) -> None:
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    epoch_iter = tqdm(range(epochs), desc="Epoch")
    for epoch_iter_idx in epoch_iter:
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="  Train", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_iter.set_postfix({"epoch": epoch_iter_idx + 1, "loss": f"{avg_loss:.4f}"})


def evaluate_model(model: BasicUNet, test_loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    dice_sum = np.zeros(3, dtype=np.float64)
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Eval", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            predicted = torch.argmax(logits, dim=1)
            target = labels[:, 0].long()

            dice_sum += compute_dice_per_class(predicted, target, num_classes=4)
            batch_count += 1

    if batch_count == 0:
        raise RuntimeError("No test batches were available for evaluation.")

    return (dice_sum / batch_count).astype(np.float32)


def train_and_eval(
    train_ids: list[str],
    test_ids: list[str],
    name: str,
    pipeline: Compose,
    device: str,
    use_amp: bool,
    epochs: int,
    batch_size: int,
    seed: int,
    train_transform: IntensityAugmentd | None = None,
) -> dict[str, object] | None:
    print("\n" + "=" * 60)
    print(f"Variant: {name}")
    print("=" * 60)

    print("Loading training data...")
    train_images, train_labels = load_vendor_slices(train_ids, pipeline)
    print("Loading test data...")
    test_images, test_labels = load_vendor_slices(test_ids, pipeline)

    if not train_images or not test_images:
        print("  [skip] Not enough data")
        return None

    train_dataset = Dataset(
        data=[{"image": image, "label": label} for image, label in zip(train_images, train_labels)],
        transform=train_transform,
    )
    test_dataset = Dataset(
        data=[{"image": image, "label": label} for image, label in zip(test_images, test_labels)],
        transform=None,
    )

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=pad_list_data_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=pad_list_data_collate,
    )

    set_seed(seed)
    model = build_model(device)
    train_model(model, train_loader, epochs=epochs, device=device, use_amp=use_amp)
    dice_per_class = evaluate_model(model, test_loader, device=device)
    avg_dice = float(np.mean(dice_per_class))

    print(
        f"  Dice: LV={dice_per_class[0]:.4f}, "
        f"RV={dice_per_class[1]:.4f}, MYO={dice_per_class[2]:.4f}"
    )
    print(f"  Avg Dice: {avg_dice:.4f}")

    return {"name": name, "dice": dice_per_class, "avg": avg_dice}


def save_results_table(results: list[dict[str, object]], save_path: Path) -> None:
    lines = ["Exp005 Intensity Augmentation Results", "=" * 58, ""]
    lines.append(f"{'Variant':<32} {'LV':<8} {'RV':<8} {'MYO':<8} {'Avg':<8}")
    lines.append("-" * 58)
    for result in results:
        dice = result["dice"]
        assert isinstance(dice, np.ndarray)
        lines.append(
            f"{result['name']:<32} "
            f"{dice[0]:<8.4f} {dice[1]:<8.4f} {dice[2]:<8.4f} {result['avg']:<8.4f}"
        )

    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Results table saved: {save_path}")


def plot_results(results: list[dict[str, object]], save_path: Path) -> None:
    names = [str(result["name"]) for result in results]
    lv = [float(result["dice"][0]) for result in results]
    rv = [float(result["dice"][1]) for result in results]
    myo = [float(result["dice"][2]) for result in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, lv, width, label="LV")
    ax.bar(x, rv, width, label="RV")
    ax.bar(x + width, myo, width, label="MYO")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.7981, color="gray", linestyle="--", alpha=0.5, label="Exp004 Mixed Avg")
    ax.legend()
    ax.set_title("Exp005 Split3: Intensity Augmentation")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Barplot saved: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp005 intensity augmentation for M&Ms vendor shift.",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs per variant.")
    parser.add_argument("--batch-size", type=int, default=8, help="2D slice batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run only the intensity-augmentation variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device, use_amp = get_device()
    print("=" * 60)
    print("Exp005: Intensity augmentation")
    print(f"Device: {device} (AMP={'ON' if use_amp else 'OFF'})")
    print("=" * 60)

    vendor_map = load_vendor_map()
    siemens = get_cases_by_vendor(vendor_map, ["Siemens"])
    philips = get_cases_by_vendor(vendor_map, ["Philips"])
    ge_canon = get_cases_by_vendor(vendor_map, ["GE", "Canon"])

    print(
        f"Siemens: {len(siemens)} cases, "
        f"Philips: {len(philips)} cases, GE+Canon: {len(ge_canon)} cases"
    )

    train_ids = siemens + philips
    test_ids = ge_canon
    pipeline = build_pipeline()
    results: list[dict[str, object]] = []

    if not args.skip_baseline:
        baseline = train_and_eval(
            train_ids=train_ids,
            test_ids=test_ids,
            name="Split3 baseline",
            pipeline=pipeline,
            device=device,
            use_amp=use_amp,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            train_transform=None,
        )
        if baseline:
            results.append(baseline)

    intensity_aug = IntensityAugmentd(
        config=IntensityAugmentConfig(),
        seed=args.seed + 1000,
    )
    augmented = train_and_eval(
        train_ids=train_ids,
        test_ids=test_ids,
        name="Split3 intensity aug",
        pipeline=pipeline,
        device=device,
        use_amp=use_amp,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        train_transform=intensity_aug,
    )
    if augmented:
        results.append(augmented)

    if results:
        save_results_table(results, OUTPUT_ROOT / "exp005_intensity_dice_table.txt")
        plot_results(results, OUTPUT_ROOT / "exp005_intensity_dice_barplot.png")

    print("\n" + "=" * 60)
    print("Exp005 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
