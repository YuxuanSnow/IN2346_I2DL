"""Define tests, sanity checks, and evaluation"""

from .image_folder_dataset_tests import test_image_folder_dataset
from .transform_tests import (
    test_rescale_transform,
    test_compute_image_mean_and_std
)
from .dataloader_tests import test_dataloader
from .eval_utils import save_pickle
