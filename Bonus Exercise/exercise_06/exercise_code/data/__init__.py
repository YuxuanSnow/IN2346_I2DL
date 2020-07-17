"""Definition of all datasets and dataloader"""

from .base_dataset import DummyDataset
from .image_folder_dataset import (
    ImageFolderDataset,
    RescaleTransform,
    NormalizeTransform,
    FlattenTransform,
    ComposeTransform,
    compute_image_mean_and_std,
)
from .dataloader import DataLoader
