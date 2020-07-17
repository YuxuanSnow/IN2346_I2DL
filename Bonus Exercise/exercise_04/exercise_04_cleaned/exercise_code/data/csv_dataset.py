from .base_dataset import Dataset
import numpy as np
import pandas as pd
import os.path


class CSVDataset(Dataset):
    """ 
    CSVDataset class.
    Provide access to the Boston Housing Prices dataset. 
    """
    def __init__(self, target_column, transform=None, mode="train", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The name of the .csv dataset file should be the same as the name
        # of the archive, but with a different extension.
        name_prefix = self.dataset_zip_name[:self.dataset_zip_name.find('.')]
        dataset_csv_name = name_prefix + '.csv'
        data_path = os.path.join(self.root_path, dataset_csv_name)

        self.target_column = target_column
        self.df = pd.read_csv(data_path)

        # split the dataset into train - val - test with the ratio 60 - 20 - 20
        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        train, val, test = np.split(self.df.sample(frac=1, random_state=0), [int(.6*len(self.df)), int(.8*len(self.df))])
        if mode == "train":
            self.df = train
        elif mode == "val":
            self.df = val
        elif mode == "test":
            self.df = test

        self.data = self.df.loc[:, self.df.columns != self.target_column]
        self.targets = self.df[self.target_column]
        self.transforms = transform if transform is not None else lambda x : x

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
