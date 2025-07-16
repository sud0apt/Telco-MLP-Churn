import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import TelcoDataset, get_cat_dims
from model import MLP
from train import train_model
import config
import torch

# Load data
df = pd.read_csv("telco-churm.csv")

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Datasets
train_dataset = TelcoDataset(train_df, config.categorical_cols, config.numerical_cols, config.target_col)
test_dataset = TelcoDataset(test_df, config.categorical_cols, config.numerical_cols, config.target_col)

# Model
cat_dims = get_cat_dims(df, config.categorical_cols)
emb_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]
model = MLP(cat_dims, emb_dims, len(config.numerical_cols), config.hidden_dims, config.output_dim)

# Train
train_model(model, train_dataset, test_dataset, config)

# Save the weights
torch.save(model.state_dict(), "mlp_telco_churn.pt")
