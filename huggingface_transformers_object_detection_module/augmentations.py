import albumentations

def get_train_transform(max_image_size=800):
    """Returns the training augmentation pipeline."""
    return albumentations.Compose(
        [
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=0.9),
                albumentations.VerticalFlip(p=0.6)
            ], p=0.8),
            albumentations.OneOf([
                albumentations.ColorJitter(p=0.5),
                albumentations.RandomBrightnessContrast(p=0.5),
                albumentations.HueSaturationValue(p=0.5),
                albumentations.CLAHE(p=0.5),
                albumentations.RandomGamma(p=0.5),
            ], p=0.8),
            albumentations.BBoxSafeRandomCrop(erosion_rate=0.5, p=0.5),
            albumentations.LongestMaxSize(max_size=max_image_size),
            albumentations.PadIfNeeded(max_image_size, max_image_size, border_mode=0, position="center"),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"], clip=True, min_visibility=0.5),
    )

def get_validation_transform(max_image_size=800):
    """Returns the validation transformation pipeline."""
    return albumentations.Compose(
        [
            albumentations.LongestMaxSize(max_size=max_image_size),
            albumentations.PadIfNeeded(max_image_size, max_image_size, border_mode=0, position="center"),
        ],
        bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"], clip=True),
    )