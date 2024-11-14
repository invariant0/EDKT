import torch
import torch.nn as nn 
import torch.nn.functional as F 

# class MLP_encoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, embed_dim, num_layer):
#         super(MLP_encoder, self).__init__()
#         self.linear_module_list = nn.ModuleList()
#         self.ln_module_list = nn.ModuleList()
#         self.linear_module_list.append(nn.Linear(input_dim, hidden_dim))
#         self.ln_module_list.append(nn.LayerNorm(hidden_dim))
#         for _ in range(num_layer - 2):
#             self.linear_module_list.append(nn.Linear(hidden_dim, hidden_dim))
#             self.ln_module_list.append(nn.LayerNorm(hidden_dim))
#         self.linear_module_list.append(nn.Linear(hidden_dim, embed_dim))
#         self.ln_module_list.append(nn.LayerNorm(embed_dim))

#     def forward(self, x):
#         out = x
#         for i in range(len(self.linear_module_list)):
#             out = self.linear_module_list[i](out)
#             out = self.ln_module_list[i](out)
#             if i < len(self.linear_module_list) - 1:
#                 out = F.leaky_relu(out)
#         return out

class MLP_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layer):
        super(MLP_encoder, self).__init__()
        self.linear_module_list = nn.ModuleList()
        self.ln_module_list = nn.ModuleList()
        self.input_proj = None
        
        # Input projection if dimensions don't match
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # First layer
        self.linear_module_list.append(nn.Linear(input_dim, hidden_dim))
        self.ln_module_list.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layer - 2):
            self.linear_module_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.ln_module_list.append(nn.LayerNorm(hidden_dim))
            
        # Output layer
        self.linear_module_list.append(nn.Linear(hidden_dim, embed_dim))
        self.ln_module_list.append(nn.LayerNorm(embed_dim))

    def forward(self, x):
        # Initial projection if needed
        if self.input_proj is not None:
            identity = self.input_proj(x)
        else:
            identity = x
            
        out = x
        
        for i in range(len(self.linear_module_list)):
            # Store residual
            residual = out if i > 0 else identity
            
            # Main branch
            out = self.linear_module_list[i](out)
            out = self.ln_module_list[i](out)
            
            # Apply activation except for last layer
            if i < len(self.linear_module_list) - 1:
                out = F.leaky_relu(out)
                
                # Add skip connection for hidden layers
                # Skip the first layer unless we have input projection
                if i > 0 or (i == 0 and self.input_proj is not None):
                    if out.shape == residual.shape:  # Only add if shapes match
                        out = out + residual
                        
        return out