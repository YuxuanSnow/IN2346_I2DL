"""Dataset Base Class"""

from abc import ABC, abstractmethod

from .download_utils import download_dataset


class Dataset(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define __getitem__() and __len__()
    """
    def __init__(self, root, download_url=None, force_download=False):
        self.root_path = root
        # The actual archive name should be all the text of the url after the
        # last '/'.
        if download_url is not None:
            dataset_zip_name = download_url[download_url.rfind('/')+1:]
            self.dataset_zip_name = dataset_zip_name
            download_dataset(
                url=download_url,
                data_dir=root,
                dataset_zip_name=dataset_zip_name,
                force_download=force_download,
            )

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""


class DummyDataset(Dataset):
    """
    Simple dummy dataset
    Contains all integers from 1 to a given limit, which are dividable by a given divisor
    """

    def __init__(self, divisor, limit, **kwargs):
        """
        :param divisor: common divisor of all integers in the dataset
        :param limit: upper limit of integers in the dataset
        """
        super().__init__(**kwargs)
        self.data = [i for i in range(1, limit + 1) if i % divisor == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"data": self.data[index]}
