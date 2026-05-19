import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


 

NUM_CLASSES_OF_FOOD101 = 10
DATASET_NAME = "ImageNette"
# ImageNette 224x224

class ImageNette224Dataset:
    """
    ImageNette 224x224 dataset.

    Classes: 10
    Split: train / val

    This wrapper first tries torchvision.datasets.Imagenette.
    If your torchvision version does not support Imagenette,
    it falls back to downloading imagenette2-320.tgz and reading it with ImageFolder.

    Recommended for:
    - small ImageNet-like architecture comparison
    - from-scratch sanity check
    - ConvNeXt / Mamba / ViT / CNN comparison
    """

    IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    ROOT = "dataset/"
    FOLDER_NAME = "imagenette2-320"

    default_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    default_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    @staticmethod
    def _download_imagenette_if_needed():

        dataset_dir = os.path.join(
            ImageNette224Dataset.ROOT,
            ImageNette224Dataset.FOLDER_NAME
        )

        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")

        if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):

            os.makedirs(ImageNette224Dataset.ROOT, exist_ok=True)

            download_and_extract_archive(
                url=ImageNette224Dataset.IMAGENETTE_URL,
                download_root=ImageNette224Dataset.ROOT,
                extract_root=ImageNette224Dataset.ROOT,
                filename="imagenette2-320.tgz",
                remove_finished=False,
            )

        return dataset_dir

 
    @staticmethod
    def _make_dataset(split, transformFunc):

        dataset_dir = ImageNette224Dataset._download_imagenette_if_needed()
        if split == "train":
            split_dir = os.path.join(dataset_dir, "train")
        elif split in ["val", "valid", "validation", "test"]:
            split_dir = os.path.join(dataset_dir, "val")
        else:
            raise ValueError(f"Unsupported ImageNette split: {split}")

        return ImageFolder(
            root=split_dir,
            transform=transformFunc,
        )

    @staticmethod
    def GetTrainLoader(batch_size, input_size=224, transformFunc=None, augment=True):
        if transformFunc is None:
            if augment:
                transformFunc = transforms.Compose([
                    transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])
            else:
                transformFunc = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ])

        train_dataset = ImageNette224Dataset._make_dataset(
            split="train",
            transformFunc=transformFunc,
        )

        train_dataset.train = True

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        return train_loader

    @staticmethod
    def GetTestLoader(batch_size, input_size=224, transformFunc=None):
        if transformFunc is None:
            transformFunc = transforms.Compose([
                transforms.Resize(int(input_size * 256 / 224)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

        test_dataset = ImageNette224Dataset._make_dataset(
            split="val",
            transformFunc=transformFunc,
        )

        test_dataset.train = False

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        return test_loader










