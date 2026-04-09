import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from config import BATCH_SIZE, DATA_DIR, IMAGE_SIZE


class CustomSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_dataloaders():
    base_dataset = datasets.ImageFolder(root=str(DATA_DIR))
    class_names = base_dataset.classes
    num_classes = len(class_names)

    _sample_img, _ = base_dataset[0]
    _w, _h = _sample_img.size
    assert _h == IMAGE_SIZE and _w == IMAGE_SIZE, "Image size mismatch!"

    train_size = int(0.8 * len(base_dataset))
    val_size = len(base_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), value=0),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = CustomSubset(train_subset, transform=train_transform)
    val_dataset = CustomSubset(val_subset, transform=val_transform)
    train_eval_dataset = CustomSubset(train_subset, transform=val_transform)
    full_dataset = CustomSubset(
        torch.utils.data.Subset(base_dataset, range(len(base_dataset))), transform=train_transform
    )

    # DataLoaders
    loaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        "train_eval": DataLoader(
            train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
        ),
        "full": DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
    }

    return loaders, class_names, num_classes
