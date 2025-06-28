# object_detection_utils/augmentations.py

import albumentations as A

def get_train_transform(max_image_size=600):
    """Returns the advanced training augmentation pipeline."""
    return A.Compose(
        [
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=max_image_size, p=1.0),
                    A.RandomSizedBBoxSafeCrop(height=max_image_size, width=max_image_size, p=1.0),
                ],
                p=0.2,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=7, p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.Defocus(radius=(1, 5), alias_blur=(0.1, 0.25), p=0.1),
                ],
                p=0.1,
            ),
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )

def get_validation_transform():
    """Returns the validation transformation pipeline (no actual augmentation)."""
    return A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )
