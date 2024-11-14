import torch 
import torch.nn as nn
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F 
from torch_geometric.nn import GATConv, GCNConv, NNConv
from torch_geometric.nn import MessagePassing

class GAT_Graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, heads):
        super(GAT_Graph, self).__init__()
        self.GAT_module_list = nn.ModuleList()
        self.LN_module_list = nn.ModuleList()
        self.GAT_module_list.append(GATConv(input_dim, hidden_dim, heads=heads))
        self.LN_module_list.append(nn.LayerNorm(hidden_dim * heads))
        for _ in range(num_layers - 1):
            self.GAT_module_list.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
            self.LN_module_list.append(nn.LayerNorm(hidden_dim * heads))
        self.linear = nn.Linear(hidden_dim*heads*2, embed_dim)
        self.num_layers = num_layers
    def forward(self, batch):
        x, edge_index, edge_attr, batch= batch.x, batch.edge_index, batch.edge_attr, batch.batch
        out = x
        for idx, (GATmodel, Layer_Norm) in enumerate(zip(self.GAT_module_list, self.LN_module_list)):
            out = GATmodel(out, edge_index, edge_attr)
            out = Layer_Norm(out)
            out = F.leaky_relu(out, negative_slope=0.1)
        out = torch.cat([gmp(out, batch), gap(out, batch)], dim=1)
        out = self.linear(out)
        return out

# class GAT_Graph(nn.Module):
#     def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, heads):
#         super(GAT_Graph, self).__init__()
#         self.GAT_module_list = nn.ModuleList()
#         self.LN_module_list = nn.ModuleList()
#         self.input_proj = None
        
#         # Input projection if dimensions don't match
#         if input_dim != hidden_dim * heads:
#             self.input_proj = nn.Linear(input_dim, hidden_dim * heads)
            
#         # First layer
#         self.GAT_module_list.append(GATConv(input_dim, hidden_dim, heads=heads))
#         self.LN_module_list.append(nn.LayerNorm(hidden_dim * heads))
        
#         # Hidden layers
#         for _ in range(num_layers - 1):
#             self.GAT_module_list.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
#             self.LN_module_list.append(nn.LayerNorm(hidden_dim * heads))
            
#         self.linear = nn.Linear(hidden_dim * heads * 2, embed_dim)
#         self.num_layers = num_layers

#     def forward(self, batch):
#         x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
#         # Initial projection if needed
#         if self.input_proj is not None:
#             identity = self.input_proj(x)
#         else:
#             identity = x
            
#         out = x
        
#         for idx, (GATmodel, Layer_Norm) in enumerate(zip(self.GAT_module_list, self.LN_module_list)):
#             # Main branch
#             residual = out
#             out = GATmodel(out, edge_index, edge_attr)
#             out = Layer_Norm(out)
#             out = F.leaky_relu(out, negative_slope=0.1)
            
#             # Skip connection
#             if idx > 0:  # Skip connection after first layer
#                 out = out + residual
#             elif idx == 0 and self.input_proj is not None:
#                 out = out + identity
                
#         out = torch.cat([gmp(out, batch), gap(out, batch)], dim=1)
#         out = self.linear(out)
#         return out

# GCN Implementation
# class GCN_Graph(nn.Module):
#     def __init__(self, input_dim, hidden_dim, embed_dim, num_layers):
#         super(GCN_Graph, self).__init__()
#         self.GCN_module_list = nn.ModuleList()
#         self.LN_module_list = nn.ModuleList()
        
#         # First layer
#         self.GCN_module_list.append(GCNConv(input_dim, hidden_dim))
#         self.LN_module_list.append(nn.LayerNorm(hidden_dim))
        
#         # Hidden layers
#         for _ in range(num_layers - 1):
#             self.GCN_module_list.append(GCNConv(hidden_dim, hidden_dim))
#             self.LN_module_list.append(nn.LayerNorm(hidden_dim))
            
#         self.linear = nn.Linear(hidden_dim * 2, embed_dim)
#         self.num_layers = num_layers

#     def forward(self, batch):
#         x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
#         out = x
        
#         for idx, (GCNmodel, Layer_Norm) in enumerate(zip(self.GCN_module_list, self.LN_module_list)):
#             out = GCNmodel(out, edge_index)
#             out = Layer_Norm(out)
#             out = F.leaky_relu(out, negative_slope=0.1)
            
#         out = torch.cat([gmp(out, batch), gap(out, batch)], dim=1)
#         out = self.linear(out)
#         return out

class GCN_Graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers):
        super(GCN_Graph, self).__init__()
        self.GCN_module_list = nn.ModuleList()
        self.LN_module_list = nn.ModuleList()
        self.input_proj = None
        
        # Input projection if input_dim != hidden_dim
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # First layer
        self.GCN_module_list.append(GCNConv(input_dim, hidden_dim))
        self.LN_module_list.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.GCN_module_list.append(GCNConv(hidden_dim, hidden_dim))
            self.LN_module_list.append(nn.LayerNorm(hidden_dim))
            
        self.linear = nn.Linear(hidden_dim * 2, embed_dim)
        self.num_layers = num_layers

    def forward(self, batch):
        x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # Initial projection if needed
        if self.input_proj is not None:
            identity = self.input_proj(x)
        else:
            identity = x
            
        out = x
        
        for idx, (GCNmodel, Layer_Norm) in enumerate(zip(self.GCN_module_list, self.LN_module_list)):
            # Main branch
            residual = out
            out = GCNmodel(out, edge_index)
            out = Layer_Norm(out)
            out = F.leaky_relu(out, negative_slope=0.1)
            
            # Skip connection
            if idx > 0:  # Skip connection after first layer
                out = out + residual
            elif idx == 0 and self.input_proj is not None:
                out = out + identity
                
        out = torch.cat([gmp(out, batch), gap(out, batch)], dim=1)
        out = self.linear(out)
        return out

# MPNN Implementation
class MPNN_Graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, edge_attr_dim):
        super(MPNN_Graph, self).__init__()
        self.MPNN_module_list = nn.ModuleList()
        self.LN_module_list = nn.ModuleList()
        
        # First layer
        self.MPNN_module_list.append(NNConv(
            input_dim, 
            hidden_dim,
            nn.Sequential(
                nn.Linear(edge_attr_dim, hidden_dim * input_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim * input_dim, hidden_dim * input_dim)
            )
        ))
        self.LN_module_list.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.MPNN_module_list.append(NNConv(
                hidden_dim,
                hidden_dim,
                nn.Sequential(
                    nn.Linear(edge_attr_dim, hidden_dim * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
                )
            ))
            self.LN_module_list.append(nn.LayerNorm(hidden_dim))
            
        self.linear = nn.Linear(hidden_dim * 2, embed_dim)
        self.num_layers = num_layers

    def forward(self, batch):
        x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        out = x
        
        for idx, (MPNNmodel, Layer_Norm) in enumerate(zip(self.MPNN_module_list, self.LN_module_list)):
            out = MPNNmodel(out, edge_index, edge_attr)
            out = Layer_Norm(out)
            out = F.leaky_relu(out, negative_slope=0.1)
            
        out = torch.cat([gmp(out, batch), gap(out, batch)], dim=1)
        out = self.linear(out)
        return out