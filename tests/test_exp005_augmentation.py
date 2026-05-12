import numpy as np

from scripts.exp005_augmentation import (
    IntensityAugmentConfig,
    IntensityAugmentd,
    augment_image_intensity,
    compute_dice_per_class,
)


def test_augment_image_intensity_preserves_shape_background_and_input():
    image = np.array(
        [[[0.0, 1.0], [2.0, 0.0]]],
        dtype=np.float32,
    )
    original = image.copy()
    rng = np.random.default_rng(7)
    config = IntensityAugmentConfig(
        prob=1.0,
        scale_range=(2.0, 2.0),
        shift_range=(0.5, 0.5),
        noise_std_range=(0.0, 0.0),
    )

    augmented = augment_image_intensity(image, rng, config)

    assert augmented.shape == image.shape
    assert augmented.dtype == np.float32
    assert np.array_equal(image, original)
    assert augmented[0, 0, 0] == 0.0
    assert augmented[0, 1, 1] == 0.0
    assert np.allclose(augmented[image != 0], image[image != 0] * 2.0 + 0.5)


def test_intensity_augmentd_keeps_label_unchanged():
    sample = {
        "image": np.array([[[0.0, 1.0], [2.0, 0.0]]], dtype=np.float32),
        "label": np.array([[[0, 1], [2, 0]]], dtype=np.int64),
    }
    transform = IntensityAugmentd(
        config=IntensityAugmentConfig(
            prob=1.0,
            scale_range=(1.0, 1.0),
            shift_range=(1.0, 1.0),
            noise_std_range=(0.0, 0.0),
        ),
        seed=3,
    )

    result = transform(sample)

    assert np.array_equal(result["label"], sample["label"])
    assert result["image"][0, 0, 0] == 0.0
    assert result["image"][0, 1, 1] == 0.0
    assert np.allclose(
        result["image"][sample["image"] != 0],
        sample["image"][sample["image"] != 0] + 1.0,
    )


def test_compute_dice_per_class_matches_known_values():
    pred = np.array([[[1, 1], [2, 0]]])
    true = np.array([[[1, 0], [2, 3]]])

    dice = compute_dice_per_class(pred, true, num_classes=4)

    assert np.allclose(dice, np.array([2 / 3, 1.0, 0.0], dtype=np.float32))
