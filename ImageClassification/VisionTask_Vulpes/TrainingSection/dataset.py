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
        return None

    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    if len(class_names) == 0:
        return None

    return len(class_names)


# 給舊的 interface / model factory 使用
# 如果 interface 會讀 NUM_CLASSES，這裡不能是 0。
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
      test/
        red_fox/
        arctic_fox/
        ...

    Current MVP behavior:
    - train split uses dataset/train
    - val split uses dataset/test
    - test split uses dataset/test

    Note:
    Since validation and testing both use dataset/test, the reported test result
    is not an independent held-out estimate if dataset/test is used for model
    selection or hyperparameter tuning.
    """

    ROOT = "dataset"

    TRAIN_SPLIT = "train"
    VAL_SPLIT = "test"
    TEST_SPLIT = "test"

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
    def _split_dir(split_name):
        return os.path.join(FoxSpeciesDataset.ROOT, split_name)

    @staticmethod
    def _list_classes(split_name):
        split_dir = FoxSpeciesDataset._split_dir(split_name)

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

        train_dir = FoxSpeciesDataset._split_dir(FoxSpeciesDataset.TRAIN_SPLIT)
        test_dir = FoxSpeciesDataset._split_dir(FoxSpeciesDataset.TEST_SPLIT)

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

    @staticmethod
    def _check_class_consistency():
        train_classes = FoxSpeciesDataset._list_classes(FoxSpeciesDataset.TRAIN_SPLIT)
        test_classes = FoxSpeciesDataset._list_classes(FoxSpeciesDataset.TEST_SPLIT)

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
        Resolves logical split name to physical folder.

        Logical split:
        - train -> dataset/train
        - val   -> dataset/test
        - test  -> dataset/test
        """
        FoxSpeciesDataset._check_dataset_root()
        FoxSpeciesDataset._check_class_consistency()

        split = split.lower()

        if split == "train":
            physical_split = FoxSpeciesDataset.TRAIN_SPLIT

        elif split in ["val", "valid", "validation"]:
            physical_split = FoxSpeciesDataset.VAL_SPLIT

        elif split == "test":
            physical_split = FoxSpeciesDataset.TEST_SPLIT

        else:
            raise ValueError(f"Unsupported split: {split}")

        return FoxSpeciesDataset._split_dir(physical_split)

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

        return DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    @staticmethod
    def GetValLoader(
        batch_size,
        input_size=224,
        transformFunc=None,
        num_workers=2,
        pin_memory=True,
    ):
        """
        Validation uses dataset/test in the current MVP setting.
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

        return DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    @staticmethod
    def GetTestLoader(
        batch_size,
        input_size=224,
        transformFunc=None,
        num_workers=2,
        pin_memory=True,
    ):
        """
        Test uses dataset/test.
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

        return DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    @staticmethod
    def GetClassNames():
        """
        Return class names from dataset/train.
        """
        return FoxSpeciesDataset._list_classes(FoxSpeciesDataset.TRAIN_SPLIT)

    @staticmethod
    def GetNumClasses():
        """
        Return number of classes from dataset/train.
        """
        return len(FoxSpeciesDataset.GetClassNames())

    @staticmethod
    def GetDatasetStats():
        """
        Return basic dataset statistics.
        """
        stats = {}

        for logical_split in ["train", "val", "test"]:
            dataset = FoxSpeciesDataset._make_dataset(
                split=logical_split,
                transformFunc=transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]),
            )

            stats[logical_split] = {
                "num_images": len(dataset),
                "classes": dataset.classes,
                "class_to_idx": dataset.class_to_idx,
                "physical_dir": FoxSpeciesDataset._resolve_split_dir(logical_split),
            }

        return stats

    @staticmethod
    def PrintDatasetInfo():
        stats = FoxSpeciesDataset.GetDatasetStats()

        print(f"Dataset name: {DATASET_NAME}")
        print(f"Root: {FoxSpeciesDataset.ROOT}")
        print(f"NUM_CLASSES: {FoxSpeciesDataset.GetNumClasses()}")
        print(f"Classes ({len(FoxSpeciesDataset.GetClassNames())}):")

        for idx, cls_name in enumerate(FoxSpeciesDataset.GetClassNames()):
            print(f"  {idx}: {cls_name}")

        print("\nSplit mapping:")
        print(f"  train -> {stats['train']['physical_dir']}")
        print(f"  val   -> {stats['val']['physical_dir']}")
        print(f"  test  -> {stats['test']['physical_dir']}")

        print("\nClass mapping:")
        print(f"  Train class_to_idx: {stats['train']['class_to_idx']}")
        print(f"  Val class_to_idx:   {stats['val']['class_to_idx']}")
        print(f"  Test class_to_idx:  {stats['test']['class_to_idx']}")

        print("\nImage counts:")
        print(f"  Train images: {stats['train']['num_images']}")
        print(f"  Val images:   {stats['val']['num_images']}  # using dataset/test")
        print(f"  Test images:  {stats['test']['num_images']} # using dataset/test")