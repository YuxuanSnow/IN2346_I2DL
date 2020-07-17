"""Dataset Base Class"""

from torch.utils.data import Dataset

from .download_utils import download_dataset


class BaseDataset(Dataset):
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
