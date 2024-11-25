import torch 
import torch.nn as nn
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import torch.nn.functional as F 
from torch_geometric.nn import GATConv, GCNConv, NNConv, GINConv, SAGEConv
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

# GCN Implementation
class GCN_Graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers):
        super(GCN_Graph, self).__init__()
        self.GCN_module_list = nn.ModuleList()
        self.LN_module_list = nn.ModuleList()
        
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
        out = x
        
        for idx, (GCNmodel, Layer_Norm) in enumerate(zip(self.GCN_module_list, self.LN_module_list)):
            out = GCNmodel(out, edge_index)
            out = Layer_Norm(out)
            out = F.leaky_relu(out, negative_slope=0.1)
            
        out = torch.cat([gmp(out, batch), gap(out, batch)], dim=1)
        out = self.linear(out)
        return out

class GIN_Graph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, eps=0.0):
        super(GIN_Graph, self).__init__()
        self.GIN_module_list = nn.ModuleList()
        self.LN_module_list = nn.ModuleList()
        
        # First layer
        mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),  # Modified
            nn.ReLU(inplace=False),  # Modified
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.GIN_module_list.append(GINConv(mlp1, eps=eps, train_eps=True))
        self.LN_module_list.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, track_running_stats=False),  # Modified
                nn.ReLU(inplace=False),  # Modified
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.GIN_module_list.append(GINConv(mlp, eps=eps, train_eps=True))
            self.LN_module_list.append(nn.LayerNorm(hidden_dim))
            
        self.linear = nn.Linear(hidden_dim * 2, embed_dim)
        self.num_layers = num_layers

    def forward(self, batch):
        x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        out = x.clone()  # Added clone to prevent in-place modifications
        
        for idx, (GINmodel, Layer_Norm) in enumerate(zip(self.GIN_module_list, self.LN_module_list)):
            out = GINmodel(out, edge_index)
            out = Layer_Norm(out)
            out = F.leaky_relu(out, negative_slope=0.1, inplace=False)  # Modified
            
        out = torch.cat([gmp(out, batch), gap(out, batch)], dim=1)
        out = self.linear(out)
        return out

# GraphSAGE implementation
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.linear = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = torch.cat([gmp(x, batch.batch), gap(x, batch.batch)], dim=1)
        return self.linear(x)