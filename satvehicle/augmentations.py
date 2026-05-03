"""
Albumentations pipelines for aerial vehicle detection.

Required transforms (assignment): rotation, flip, brightness/contrast, scaling.
Use together with YOLO training (offline export or custom dataset) — see README.
"""

from __future__ import annotations

import cv2
import albumentations as A


def get_train_augmentation(
    image_size: int = 1024,
    max_rotate_degrees: int = 180,
    apply_bbox: bool = True,
) -> A.Compose:
    """
    Training augmentation with bounding-box safe geometric transforms.

    ``max_rotate_degrees=180`` covers full orientation sampling (pair with random flip).
    """
    bbox_params = (
        A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3)
        if apply_bbox
        else None
    )
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.Affine(
                rotate=(-max_rotate_degrees, max_rotate_degrees),
                scale=(0.75, 1.25),
                translate_percent=(0.02, 0.02),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.9,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.Resize(image_size, image_size),
        ],
        bbox_params=bbox_params,
    )


def get_val_augmentation(image_size: int = 1024) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size)])


def get_train_augmentation_no_bbox(image_size: int = 1024) -> A.Compose:
    """For visualization / smoke tests without labels."""
    return A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.Affine(
                rotate=(-180, 180),
                scale=(0.75, 1.25),
                translate_percent=(0.02, 0.02),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.9,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            A.Resize(image_size, image_size),
        ],
    )
