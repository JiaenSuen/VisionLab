import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASET_NAME = "FoxSpecies"


def _count_classes_from_train(root="dataset"):
    train_dir = os.path.join(root, "train")

    if not os.path.isdir(train_dir):
        # 這裡不要直接 raise，避免某些程式 import dataset 檔案時資料夾還沒準備好
        return None

    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    if len(class_names) == 0:
        return None

    return len(class_names)


# 給舊的 interface / model factory 使用
# 如果你的 interface 會讀 NUM_CLASSES，這裡就不能是 0。
NUM_CLASSES = _count_classes_from_train("dataset")


class FoxSpeciesDataset:
    """
    Custom fox species classification dataset.

    Expected folder structure:

    dataset/
      train/
        red_fox/
        arctic_fox/
        ...
      val/
        red_fox/
        arctic_fox/
        ...
      test/
        red_fox/
        arctic_fox/
        ...

    Current MVP behavior:
    - train split uses dataset/train
    - val split uses dataset/test
    - test split uses dataset/test

    This means dataset/test is used as both validation and testing set.
    """

    ROOT = "dataset"

    default_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    default_eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    @staticmethod
    def _list_classes(split):
        split_dir = os.path.join(FoxSpeciesDataset.ROOT, split)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        classes = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])

        if len(classes) == 0:
            raise RuntimeError(f"No class folders found in: {split_dir}")

        return classes

    @staticmethod
    def _check_dataset_root():
        if not os.path.isdir(FoxSpeciesDataset.ROOT):
            raise FileNotFoundError(
                f"Dataset root not found: {FoxSpeciesDataset.ROOT}"
            )

        train_dir = os.path.join(FoxSpeciesDataset.ROOT, "train")
        test_dir = os.path.join(FoxSpeciesDataset.ROOT, "test")

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Train directory not found: {train_dir}"
            )

        if not os.path.isdir(test_dir):
            raise FileNotFoundError(
                f"Test directory not found: {test_dir}"
            )

    @staticmethod
    def _check_class_consistency():
        train_classes = FoxSpeciesDataset._list_classes("train")
        test_classes = FoxSpeciesDataset._list_classes("test")

        if train_classes != test_classes:
            raise RuntimeError(
                "Class folder mismatch between dataset/train and dataset/test.\n"
                f"Train classes: {train_classes}\n"
                f"Test classes:  {test_classes}\n"
                f"Only in train: {sorted(set(train_classes) - set(test_classes))}\n"
                f"Only in test:  {sorted(set(test_classes) - set(train_classes))}"
            )

    @staticmethod
    def _resolve_split_dir(split):
        """
        MVP setting:
        - train -> dataset/train
        - val / valid / validation -> dataset/test
        - test -> dataset/test
        """
        FoxSpeciesDataset._check_dataset_root()
        FoxSpeciesDataset._check_class_consistency()

        split = split.lower()

        if split == "train":
            return os.path.join(FoxSpeciesDataset.ROOT, "train")

        elif split in ["val", "valid", "validation", "test"]:
            return os.path.join(FoxSpeciesDataset.ROOT, "test")

        else:
            raise ValueError(f"Unsupported split: {split}")

    @staticmethod
    def _make_dataset(split, transformFunc):
        split_dir = FoxSpeciesDataset._resolve_split_dir(split)

        dataset = ImageFolder(
            root=split_dir,
            transform=transformFunc,
        )

        if len(dataset.classes) == 0:
            raise RuntimeError(f"No classes found in: {split_dir}")

        return dataset

    @staticmethod
    def GetTrainLoader(
        batch_size,
        input_size=224,
        transformFunc=None,
        augment=True,
        num_workers=2,
        pin_memory=True,
    ):
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

        train_dataset = FoxSpeciesDataset._make_dataset(
            split="train",
            transformFunc=transformFunc,
        )

        train_dataset.train = True

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader

    @staticmethod
    def GetValLoader(
        batch_size,
        input_size=224,
        transformFunc=None,
        num_workers=2,
        pin_memory=True,
    ):
        """
        In current MVP setting, validation uses dataset/test.
        """
        if transformFunc is None:
            transformFunc = transforms.Compose([
                transforms.Resize(int(input_size * 256 / 224)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

        val_dataset = FoxSpeciesDataset._make_dataset(
            split="val",
            transformFunc=transformFunc,
        )

        val_dataset.train = False

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return val_loader

    @staticmethod
    def GetTestLoader(
        batch_size,
        input_size=224,
        transformFunc=None,
        num_workers=2,
        pin_memory=True,
    ):
        """
        Test also uses dataset/test.
        """
        if transformFunc is None:
            transformFunc = transforms.Compose([
                transforms.Resize(int(input_size * 256 / 224)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

        test_dataset = FoxSpeciesDataset._make_dataset(
            split="test",
            transformFunc=transformFunc,
        )

        test_dataset.train = False

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return test_loader

    @staticmethod
    def GetClassNames():
        """
        Return class names from dataset/train.
        """
        return FoxSpeciesDataset._list_classes("train")

    @staticmethod
    def GetNumClasses():
        """
        Return number of classes from dataset/train.
        """
        return len(FoxSpeciesDataset.GetClassNames())

    @staticmethod
    def PrintDatasetInfo():
        train_dataset = FoxSpeciesDataset._make_dataset(
            split="train",
            transformFunc=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        )

        test_dataset = FoxSpeciesDataset._make_dataset(
            split="test",
            transformFunc=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        )

        print(f"Dataset name: {DATASET_NAME}")
        print(f"Root: {FoxSpeciesDataset.ROOT}")
        print(f"NUM_CLASSES: {FoxSpeciesDataset.GetNumClasses()}")
        print(f"Classes ({len(train_dataset.classes)}):")
        for idx, cls_name in enumerate(train_dataset.classes):
            print(f"  {idx}: {cls_name}")

        print(f"Train class_to_idx: {train_dataset.class_to_idx}")
        print(f"Test class_to_idx:  {test_dataset.class_to_idx}")
        print(f"Train images: {len(train_dataset)}")
        print(f"Val images: {len(test_dataset)}  # using dataset/test")
        print(f"Test images: {len(test_dataset)} # using dataset/test")