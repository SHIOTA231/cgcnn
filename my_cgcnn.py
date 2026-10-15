import torch
import torch.nn as nn

class CGConv_v4(nn.Module):
    """
    eq.4
    self.D_v = D_v
    self.D_u = D_u
    
    """
    def __init__(self, D_v, D_u):
        super(CGConv_v4, self).__init__()
        self.D_v = D_v
        self.D_u = D_u
        
        # v_j (D_v) + u (D_u) -> D_v
        self.fc_c = nn.Linear(self.D_v + self.D_u, self.D_v, bias=False)
        
        # v_i (D_v) -> D_v
        self.fc_s = nn.Linear(self.D_v, self.D_v, bias=False)
        
        # b
        self.bias = nn.Parameter(torch.zeros(self.D_v))
        
        # g (Softplus)
        self.softplus = nn.Softplus()

    def forward(self, v_in_fea, u_fea, edge_idx):
        """
        Parameters
        ----------
        v_in_fea : torch.Tensor (N_total x D_v)
        u_fea    : torch.Tensor (E_total x D_u)
        edge_idx    : torch.LongTensor (2 x E_total)                 一行目が送り手、二行目が受け手
        """
        i_idx, j_idx = edge_idx[0], edge_idx[1]                      #iを０jを１という風においた。
        # v_j
        v_j_fea = v_in_fea[j_idx]  # E_total x D_v
        # 2. v_j ⊕ u 
        z_msg = torch.cat([v_j_fea, u_fea], dim=1)     #(E_total x (D_v + D_u))
        # 3. (v_j ⊕ u) W_c
        msg_transformed = self.fc_c(z_msg)      #(E_total x D_v)
        # 4. エッジごとのメッセージを受信原子 i ごとに集約 (∑_{j,k})
        # (N_total x D_v のゼロ行列に index_add_)
        aggregated_msg = torch.zeros_like(v_in_fea)
        aggregated_msg.index_add_(0, i_idx, msg_transformed)    #Σjk
        # 5. v_i W_s を計算 (N_total x D_v)
        self_transformed = self.fc_s(v_in_fea)
        
        # 6. 集約メッセージ + 自己更新項 + バイアス b
        total_in = aggregated_msg + self_transformed + self.bias
        
        # 7. 活性化関数 g を適用
        atom_out_fea = self.softplus(total_in)
        
        return atom_out_fea
class CGConv_v5(nn.Module):
    
    def __init__(self, D_v, D_u):
        super(CGConv_v5, self).__init__()
        self.D_v = D_v
        self.D_u = D_u
        # Concatenated length: v_i (D_v) + v_j (D_v) + u_(i,j)k (D_u)
        total_in_len = 2 * self.D_v + self.D_u
        
        # Linear layers for Filter/Gate (W_f, b_f) and Core/Softplus (W_s, b_s)
        self.fc_f = nn.Linear(total_in_len, self.D_v) # Gate path W_f, b_f
        self.fc_s = nn.Linear(total_in_len, self.D_v) # Core path W_s, b_s
        
        # Activation functions: Sigmoid (\sigma) and Softplus (g)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
    def forward(self, v_in_fea, u_fea, edge_idx):
        i_idx, j_idx = edge_idx[0], edge_idx[1]
        
        # 1. Extract source and target node features
        v_i_fea = v_in_fea[i_idx]  # Shape: E_total x D_v
        v_j_fea = v_in_fea[j_idx]  # Shape: E_total x D_v
        
        # 2. Concatenate features z = [v_i | v_j | u]
        z = torch.cat([v_i_fea, v_j_fea, u_fea], dim=1)  # Shape: E_total x (2D_v + D_u)
        
        # 3. Calculate Gate path \sigma(z W_f + b_f) and Core path g(z W_s + b_s)
        gate_out = self.sigmoid(self.fc_f(z))   # Shape: E_total x D_v
        core_out = self.softplus(self.fc_s(z))  # Shape: E_total x D_v
        # 4. Element-wise product (\odot) for edge messages
        message = gate_out * core_out  # Shape: E_total x D_v
        # 5. Aggregate messages via \sum_{j,k} to target nodes i
        aggregated_msg = torch.zeros_like(v_in_fea)
        aggregated_msg.index_add_(0, i_idx, message)  # Shape: N_total x D_v
        
        # 6. Apply Residual Connection: v_i^{(t+1)} = v_i^{(t)} + aggregated_msg
        atom_out_fea = v_in_fea + aggregated_msg
        
        return atom_out_fea

class CGConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network
    """
    def __init__(self, orig_D_v, D_u, D_v=64, t_conv=3, hidden_fea_llen=128, L_hidden=1, out_dim=1, classification=False, version='v5'):   # out_dim: target dimension or num classes
        super(CGConvNet, self).__init__()
        self.version = version
        self.L_hidden = L_hidden
        self.classification = classification
        
        # 1. Embedding layer for raw atom features
        self.embedding = nn.Linear(orig_D_v, D_v)
        
        # 2. CGConv layers (v4 or v5)
        self.convs = nn.ModuleList()
        for _ in range(t_conv):
            if version == 'v4':
                self.convs.append(CGConv_v4(D_v, D_u))
            elif version == 'v5':
                self.convs.append(CGConv_v5(D_v, D_u))

        # 3. MLP layers after Pooling
        self.conv_to_fc = nn.Linear(D_v, hidden_fea_llen)      # Entry layer: D_v -> hidden_fea_llen
        # Multiple hidden layers if L_hidden > 1
        if self.L_hidden > 1:
            self.fcs = nn.ModuleList([nn.Linear(hidden_fea_llen, hidden_fea_llen) for _ in range(self.L_hidden - 1)])
        # Output layer: hidden_fea_llen -> out_dim (B x out_dim)
        self.fc_out = nn.Linear(hidden_fea_llen, out_dim)               # If classification=True, out_dim usually represents the number of classes (e.g., 2)
        
        self.softplus = nn.Softplus()

    def forward(self, v_fea, u_fea, edge_idx, B_idx):
        """
        Forward pass for CGConvNet
        """
        # Step 1: Embed raw atom features
        v_fea = self.embedding(v_fea)
    
        # Step 2: Pass through CGConv layers
        for conv in self.convs:
            v_fea = conv(v_fea, u_fea, edge_idx)
            
        # Step 3: Pooling (Mean Pooling for each crystal in batch)
        v_c_fea_list = [torch.mean(v_fea[idx], dim=0, keepdim=True) for idx in B_idx]
        v_c_fea = torch.cat(v_c_fea_list, dim=0)                # Shape: B x D_v
        
        # Step 4: MLP layers for prediction
        v_c_fea = self.softplus(self.conv_to_fc(v_c_fea))     # Shape: B x hidden_fea_llen
        
        if self.L_hidden > 1:
            for fc in self.fcs:
                v_c_fea = self.softplus(fc(v_c_fea))
                
        out = self.fc_out(v_c_fea)                             # Shape: B x out_dim
            
        return out