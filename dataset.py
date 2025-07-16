import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config
import numpy as np

class TelcoDataset(Dataset):
    def __init__(self, df, cat_cols, num_cols, target_col):
        df = df.copy()
        
        # encoding the categorical cols
        self.label_encoders = {
            col: LabelEncoder().fit(df[col]) for col in cat_cols # fit() function gets the distinct classes
        }
        
        # transform maps the original category to its integer code (series of integers)
        df[cat_cols] = df[cat_cols].apply(lambda col: self.label_encoders[col.name].transform(col))
        
        # convert to numerical
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=num_cols, inplace=True)

            
        print("Data in df[num_cols]:")
        print(df[num_cols])

        print("\nData types for each numeric column:")
        print(df[num_cols].dtypes)
        
        # calling .fit() computes the sample mean & std dev for each numerical cols
        self.scaler = StandardScaler().fit(df[num_cols])

        
        assert not np.isnan(df[num_cols]).any().any(), "NaNs in numerical features!"
        assert np.isfinite(df[num_cols]).all().all(), "Infinite values detected!"


        # scales by --> x = (x - avg from fit) / standard dev
        df[num_cols] = self.scaler.transform(df[num_cols])
        
        df[target_col] = df[target_col].map({'No': 0, 'Yes': 1})

        # convert cols to tensor
        self.cats = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.nums = torch.tensor(df[num_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.cats[idx], self.nums[idx], self.labels[idx] 
    
def get_cat_dims(df, cat_cols):
    return [df[col].nunique() for col in cat_cols]

if __name__ == "__main__":
    df = pd.read_csv('telco-churm.csv')
    dataset = TelcoDataset(df, config.categorical_cols, config.numerical_cols, config.target_col)
    print(len(dataset))

    cat_dims = get_cat_dims(df, config.categorical_cols)
    print("Unique classes per categorical column:")
    for col, dim in zip(config.categorical_cols, cat_dims):
        print(f"  - {col:20}: {dim}")