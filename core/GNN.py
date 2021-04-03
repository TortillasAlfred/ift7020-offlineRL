class GCNPolicy(torch.nn.Module):
    def __init__(self, nbrConvLayer = 1, nbrLinLayer = 1, emb_size = 64):
        super().__init__()
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19
        
        # CONSTRAINT EMBEDDING        
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
        )
        self.cons_lin = torch.nn.ModuleList()
        for _ in range(nbrLinLayer-1):
            self.cons_lin.append(torch.nn.Linear(emb_size, emb_size))

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
        )        
        self.var_lin = torch.nn.ModuleList()
        for _ in range(nbrLinLayer-1):
            self.var_lin.append(torch.nn.Linear(emb_size, emb_size))
        
        self.conv_v_to_c = torch.nn.ModuleList()
        self.conv_c_to_v = torch.nn.ModuleList()
        
        for _ in range(nbrConvLayer):
            self.conv_v_to_c.append(BipartiteGraphConvolution(emb_size))
            self.conv_c_to_v.append(BipartiteGraphConvolution(emb_size))

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        for (cons_lin, var_lin) in zip(self.cons_lin, self.var_lin):
            constraint_features = F.relu(cons_lin(constraint_features))
            variable_features = F.relu(var_lin(variable_features))
            

        # Half convolutions
        for (v_to_c, c_to_v) in zip(self.conv_v_to_c, self.conv_c_to_v):
            constraint_features = v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
            variable_features = c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output
    
    
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need 
    to provide the exact form of the messages being passed.
    """
    def __init__(self, emb_size = 64):
        super().__init__('add')
        
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )
        
        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i) 
                                           + self.feature_module_edge(edge_features) 
                                           + self.feature_module_right(node_features_j))
        return output
    

policy = GCNPolicy(nbrConvLayer = 2, nbrLinLayer = 2, emb_size = 32).to(DEVICE)
