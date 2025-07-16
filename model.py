import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Inputs:
        -cat dims: categorical col dimensions
        -emb_dims: embedding dimensions
        -num_inpit_dim: numerical col dimensions
        -hidden_dims: hidden layer dimensions
        -output_dims: output dimensions (should be 2 since binary classification)
    """
    def __init__ (self, cat_dims, emb_dims, num_input_dim, hidden_dims, output_dim):
        super().__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cat_size, 
                         embedding_dim=emb_size)
            for cat_size, emb_size in zip(cat_dims, emb_dims) # cat_size is the number of unique entries
        ])
        
        self.emb_out_dim = sum(emb_dims) # total size of the concat embedding vector across all the cat features
        
        self.fc1 = nn.Linear(self.emb_out_dim + num_input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x_cat, x_num):
        """
        Input: 
            - x_cat = [batch_size, num_categorical_columns]
            - x_num = [batch_size, num_numerical_columns]
        """
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, dim=1)
        x = torch.cat([x, x_num], dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x