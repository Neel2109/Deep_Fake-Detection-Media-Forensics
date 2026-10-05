"""The current Xception RGB resize and ImageNet normalization contract."""

IMAGE_SIZE = 299
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)


def preprocess_transform(torchvision_transforms):
    return torchvision_transforms.Compose([
        torchvision_transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision_transforms.ToTensor(),
        torchvision_transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])
