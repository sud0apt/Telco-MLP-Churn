import pandas as pd

df = pd.read_csv('telco-churm.csv')

target_col = 'Churn'
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
drop_cols = ['customerID', 'Churn']
categorical_cols = []
    
for colname in df.columns:
    if colname not in numerical_cols and colname not in drop_cols:
        categorical_cols.append(colname)
        
        
hidden_dims = [128, 64, 32]
output_dim = 2 # binary (Churn or not)

batch_size = 64
lr = 0.001
epochs = 50
        

# print(len(categorical_cols))

# print(f"Numerical columns ({len(numerical_cols)}):")
# for col in numerical_cols:
#     print("   -", col)

# print(f"\nCategorical columns ({len(categorical_cols)}):")
# for col in categorical_cols:
#     print("   -", col)