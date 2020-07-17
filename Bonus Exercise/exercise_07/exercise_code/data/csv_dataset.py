from .base_dataset import Dataset
import numpy as np
import pandas as pd
import os.path


class CSVDataset(Dataset):
    """
    CSVDataset class.
    Provide access to the Boston Housing Prices dataset.
    """

    def __init__(self, target_column, transform=None, mode="train", input_data=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The name of the .csv dataset file should be the same as the name
        # of the archive, but with a different extension.
        if input_data is not None:
            self.df = input_data
        else:
            name_prefix = self.dataset_zip_name[:self.dataset_zip_name.find('.')]
            dataset_csv_name = name_prefix + '.csv'
            data_path = os.path.join(self.root_path, dataset_csv_name)
            self.df = pd.read_csv(data_path)

        self.target_column = target_column

        # split the dataset into train - val - test with the ratio 60 - 20 - 20
        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        train, val, test = np.split(self.df.sample(frac=1, random_state=0), [
                                    int(.6 * len(self.df)), int(.8 * len(self.df))])
        if mode == "train":
            self.df = train
        elif mode == "val":
            self.df = val
        elif mode == "test":
            self.df = test

        self.data = self.df.loc[:, self.df.columns != self.target_column]
        self.targets = self.df[self.target_column]
        self.transforms = transform if transform is not None else lambda x: x

        self.data.iloc[0]['OverallQual'] = np.nan

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Create a dict of the data at the given index in your dataset.

        The dict should have the following format:
        { "features" : <i-th row of the dataframe (except TARGET_COLUMN)>,
             "label" : <value of TARGET_COLUMN for i-th row> }
        """

        data_dict = {}
        data_dict['features'] = self.data.iloc[index]
        data_dict['target'] = self.targets.iloc[index]

        return self.transforms(data_dict)


class FeatureSelectorAndNormalizationTransform:
    """
    Select some numerical features and normalize them between 0 and 1.
    """

    def __init__(self, column_stats, target_column):
        """
        :param column_stats: a dictionary mapping the column name to the
            relevant statistics for normalization (min and max on that column).
            It should also include the statistics for the target column.
        """
        self.column_stats = column_stats
        self.target_column = target_column

    def __call__(self, data_dict):
        def normalize_column(old_value, column_name):
            mn = self.column_stats[column_name]['min']
            mx = self.column_stats[column_name]['max']
            return (old_value - mn) / (mx - mn)

        # For every feature column, normalize it if it's one of the columns
        # we want to keep.
        feature_columns = []
        for column_idx in data_dict['features'].index:
            if column_idx in self.column_stats and column_idx != self.target_column:
                feature_columns.append(column_idx)

                if np.isnan(data_dict['features'][column_idx]):
                    mean_col_val = self.column_stats[column_idx]['mean']
                    data_dict['features'][column_idx] = mean_col_val

                old_value = data_dict['features'][column_idx]
                normalized = normalize_column(old_value, column_idx)
                data_dict['features'][column_idx] = normalized

        # Drop the rest of the columns.
        data_dict['features'] = data_dict['features'][feature_columns]
        data_dict['features'] = data_dict['features'].values.astype(np.float32)

        # Also normalize the target.
        old_value = data_dict['target']
        normalized = normalize_column(old_value, self.target_column)
        data_dict['target'] = np.array([normalized])

        return data_dict


class FeatureSelectorTransform:
    """
    Select some numerical features and not normalize them, just return their old values.
    This class is used for the binarized data to convert it to the correct format of CSVDataset object
    so that it could be loaded by our dataloader
    """

    def __init__(self, column_stats, target_column):
        """
        :param column_stats: a dictionary mapping the column name to the
            relevant statistics for normalization (min and max on that column).
            It should also include the statistics for the target column.
        """
        self.column_stats = column_stats
        self.target_column = target_column

    def __call__(self, data_dict):

        # For every feature column, just keep it old values

        feature_columns = []
        for column_idx in data_dict['features'].index:
            if column_idx in self.column_stats and column_idx != self.target_column:
                feature_columns.append(column_idx)

                if np.isnan(data_dict['features'][column_idx]):
                    mean_col_val = self.column_stats[column_idx]['mean']
                    data_dict['features'][column_idx] = mean_col_val

        data_dict['features'] = data_dict['features'][feature_columns]
        data_dict['features'] = data_dict['features'].values.astype(np.float32)

        data_dict['target'] = np.array([data_dict['target']])

        return data_dict
    
    
def get_exercise5_transform():
    # dataloading and preprocessing steps as in ex04 2_logistic_regression.ipynb
    target_column = 'SalePrice'
    i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
    root_path = os.path.join(i2dl_exercises_path, "datasets", 'housing')
    housing_file_path = os.path.join(root_path, "housing_train.csv")
    download_url = 'https://cdn3.vision.in.tum.de/~dl4cv/housing_train.zip'

    # Always make sure this line was run at least once before trying to
    # access the data manually, as the data is downloaded in the
    # constructor of CSVDataset.
    train_dataset = CSVDataset(target_column=target_column, root=root_path, download_url=download_url, mode="train")
    
    #For the data transformations, compute min, max and mean for each feature column. We perform the same transformation
    # on the training, validation, and test data.
    df = train_dataset.df
    # Select only 2 features to keep plus the target column.
    selected_columns = ['OverallQual', 'GrLivArea', target_column]
    #selected_columns = ['GrLivArea', target_column]
    mn, mx, mean = df.min(), df.max(), df.mean()

    column_stats = {}
    for column in selected_columns:
        crt_col_stats = {'min' : mn[column],
                         'max' : mx[column],
                         'mean': mean[column]}
        column_stats[column] = crt_col_stats

    transform = FeatureSelectorAndNormalizationTransform(column_stats, target_column)
    
    return transform
