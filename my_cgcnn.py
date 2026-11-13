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
        self.fc_c = nn.Linear(self.D_v + self.D_u, self.D_v, bias=False)        #1. v_j (D_v) + u (D_u) -> D_v
        self.fc_s = nn.Linear(self.D_v, self.D_v, bias=False)                   #2. v_i (D_v) -> D_v
        self.bias = nn.Parameter(torch.zeros(self.D_v))                         #3. b (D_v)
        self.softplus = nn.Softplus()
        self.cache = {}

    def forward(self, v_in_fea, u_fea, edge_idx):
        """
        Parameters
        ----------
        v_in_fea : torch.Tensor (N_total x D_v)
        u_fea    : torch.Tensor (E_total x D_u)
        edge_idx    : torch.LongTensor (2 x E_total)                 一行目が送り手、二行目が受け手
        """
        i_idx, j_idx = edge_idx[0], edge_idx[1]                                 #4. iを０jを１という風においた。
        v_j_fea = v_in_fea[j_idx]                                               #5. E_total x D_v
        z_msg = torch.cat([v_j_fea, u_fea], dim=1)                              #6. (E_total x (D_v + D_u)) v_j ⊕ u
        msg_transformed = self.fc_c(z_msg)                                      #7. (E_total x D_v) z*=(v_j ⊕ u) W_c
        aggregated_msg = torch.zeros_like(v_in_fea)                             #8. (N_total x D_v) 受信原子ごとに集約されたメッセージのゼロ行列作成
        aggregated_msg.index_add_(0, i_idx, msg_transformed)                    #9. Σjk
        self_transformed = self.fc_s(v_in_fea)                                  #10. (N_total x D_v) v_i W_s
        total_in = aggregated_msg + self_transformed + self.bias                #11. mij                
        atom_out_fea = self.softplus(total_in)                                  #12. softplus(mij)
        self.cache['z_msg'] = z_msg
        self.cache['total_in'] = total_in
        
        return atom_out_fea

    def manual_backward(self, d_atom_out_fea, v_in_fea, u_fea, edge_idx):
        """CGConv_v4 の手動逆伝播

        引数:
            d_atom_out_fea: 出力勾配 (N_total x D_v)
            v_in_fea: 入力原子特徴量 (N_total x D_v) u_fea: ボンド特徴量 (E_total x
            D_u) edge_idx: エッジインデックス (2 x E_total)

        返り値:
            d_v_in_fea: 入力原子特徴量に対する勾配 (N_total x D_v)
            conv_grads: パラメータ勾配辞書
        """
        i_idx, j_idx = edge_idx[0], edge_idx[1]
        z_msg = self.cache['z_msg']
        total_in = self.cache['total_in']
        d_total_in = d_atom_out_fea * torch.sigmoid(total_in)                     #13. dy/dx = Sigmoid(x) Shape: N_total x D_v                            
        d_aggregated_msg = d_total_in                                           #14. Shape: N_total x D_v
        d_self_transformed = d_total_in                                         #15. Shape: N_total x D_v
        grad_bias = torch.sum(d_total_in, dim=0)                                #16. Shape: D_v
        grad_w_s = torch.matmul(d_self_transformed.T, v_in_fea)                 #17. Shape: D_v x D_v
        d_v_in_from_s = torch.matmul(d_self_transformed, self.fc_s.weight)      #18. Shape: N_total x D_v
        d_msg_transformed = d_aggregated_msg[i_idx]                             #19. Shape: E_total x D_v
        grad_w_c = torch.matmul(d_msg_transformed.T, z_msg)                     #20. Shape: D_v x (D_v + D_u)
        d_z_msg = torch.matmul(d_msg_transformed, self.fc_c.weight)             #21. Shape: E_total x (D_v + D_u)
        d_v_j_fea = d_z_msg[:, : self.D_v]                                      #22. Shape: E_total x D_v
        d_v_in_from_j = torch.zeros_like(v_in_fea)
        d_v_in_from_j.index_add_(0, j_idx, d_v_j_fea)                           #23. Shape: N_total x D_v
        d_v_in_fea = d_v_in_from_s + d_v_in_from_j                              #24. Shape: N_total x D_v
        conv_grads = {'fc_c.weight': grad_w_c,'fc_s.weight': grad_w_s,'bias': grad_bias,}
        return d_v_in_fea, conv_grads   
        
class CGConv_v5(nn.Module):
    
    def __init__(self, D_v, D_u):
        super(CGConv_v5, self).__init__()
        self.D_v = D_v
        self.D_u = D_u
        total_in_len = 2 * self.D_v + self.D_u
        self.fc_f = nn.Linear(total_in_len, self.D_v)                           # Linear layers for Filter/Gate (W_f, b_f) and Core/Softplus (W_s, b_s)Gate path W_f, b_f
        self.fc_s = nn.Linear(total_in_len, self.D_v)                           # Core path W_s, b_s
        self.sigmoid = nn.Sigmoid()                                             # Activation functions: Sigmoid (\sigma) and Softplus (g)
        self.softplus = nn.Softplus()
        self.cache = {}

    def forward(self, v_in_fea, u_fea, edge_idx):
        i_idx, j_idx = edge_idx[0], edge_idx[1]
        v_i_fea = v_in_fea[i_idx]                                               # Shape: E_total x D_v
        v_j_fea = v_in_fea[j_idx]                                               # Shape: E_total x D_v
        z = torch.cat([v_i_fea, v_j_fea, u_fea], dim=1)                         # Shape: E_total x (2D_v + D_u)
        gate_out = self.sigmoid(self.fc_f(z))                                   # Shape: E_total x D_v
        core_out = self.softplus(self.fc_s(z))                                  # Shape: E_total x D_v
        message = gate_out * core_out                                           # Shape: E_total x D_v i,j要素ごとの掛け算
        aggregated_msg = torch.zeros_like(v_in_fea)                             # 5. Aggregate messages via \sum_{j,k} to target nodes i
        aggregated_msg.index_add_(0, i_idx, message)                            # Shape: N_total x D_v
        atom_out_fea = v_in_fea + aggregated_msg                                # 6. Apply Residual Connection: v_i^{(t+1)} = v_i^{(t)} + aggregated_msg
        f_z = self.fc_f(z)                                                      # Gate の全結合出力 (Sigmoid直前)
        s_z = self.fc_s(z)                                                      # Core の全結合出力 (Softplus直前)
        self.cache['z'] = z
        self.cache['f_z'] = f_z
        self.cache['s_z'] = s_z
        self.cache['gate_out'] = gate_out
        self.cache['core_out'] = core_out
        return atom_out_fea

    def manual_backward(self, d_atom_out_fea, v_in_fea, u_fea, edge_idx):
        """1層分の CGConv の手動逆伝播

        引数:
            d_atom_out_fea: このConv層の出力に対する勾配 (N_total x D_v)
            v_in_fea: 入力原子特徴量 (N_total x D_v) u_fea: ボンド特徴量 (E_total x
            D_u) edge_idx: エッジインデックス (2 x E_total)

        返り値:
            d_v_in_fea: 1つ手前の層に渡す入力原子特徴量の勾配 (N_total x D_v)
            conv_grads: この層のパラメータ勾配辞書
        """
        i_idx, j_idx = edge_idx[0], edge_idx[1]
        z = self.cache['z']
        f_z = self.cache['f_z']
        s_z = self.cache['s_z']
        gate_out = self.cache['gate_out']
        core_out = self.cache['core_out']
        d_message = d_atom_out_fea[i_idx]                                       # Shape: E_total x D_v
        d_gate_out = d_message * core_out                                       # Shape: E_total x D_v δ_f
        d_core_out = d_message * gate_out                                       # Shape: E_total x D_v δ_c
        d_s_z = d_core_out * torch.sigmoid(s_z)                                 # Shape: E_total x D_v
        d_f_z = d_gate_out * gate_out * (1.0 - gate_out)                        # Shape: E_total x D_v
        grad_w_s = torch.matmul(d_s_z.T, z)                                     # Shape: D_v x (2D_v + D_u)
        grad_b_s = torch.sum(d_s_z, dim=0)                                      # Shape: D_v
        grad_w_f = torch.matmul(d_f_z.T, z)                                     # Shape: D_v x (2D_v + D_u)
        grad_b_f = torch.sum(d_f_z, dim=0)                                      # Shape: D_v
        # 入力 z に対する勾配 (dL/dz = \delta * W)
        d_z_core = torch.matmul(d_s_z, self.fc_s.weight)
        d_z_gate = torch.matmul(d_f_z, self.fc_f.weight)
        d_z = d_z_core + d_z_gate                                               # Shape: E_total x (2D_v + D_u)
        d_v_i_fea = d_z[:, : self.D_v]                                          # Shape: E_total x D_v          #0~Dv列をスライスして、d_v_i_feaに代入
        d_v_j_fea = d_z[:, self.D_v : 2 * self.D_v]                             # Shape: E_total x D_v         #Dv~2*Dv列までをスライスして、d_v_j_feaに代入
        d_v_in_fea = d_atom_out_fea.clone()
        d_v_in_fea.index_add_(0, i_idx, d_v_i_fea)
        d_v_in_fea.index_add_(0, j_idx, d_v_j_fea)
        conv_grads = {
            'fc_f.weight': grad_w_f,
            'fc_f.bias': grad_b_f,
            'fc_s.weight': grad_w_s,
            'fc_s.bias': grad_b_s,
        }

        return d_v_in_fea, conv_grads

class CGConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network
    """
    def __init__(self, orig_D_v, D_u, D_v=64, t_conv=3, hidden_fea_len=128, L_hidden=1, out_dim=2, classification=False, version='v5'):   # out_dim: target dimension or num classes
        super(CGConvNet, self).__init__()
        self.version = version
        self.L_hidden = L_hidden
        self.classification = classification
        self.cache = {}                                                         # 順伝播の中間変数を保存するよう
        self.grads = {}                                                         # 手計算した勾配を保存するよう
        self.embedding = nn.Linear(orig_D_v, D_v)
        self.convs = nn.ModuleList()
        for _ in range(t_conv):                                                 #self.convsというリストにt_convだけオブジェクトが追加。
            if version == 'v4':
                self.convs.append(CGConv_v4(D_v, D_u))
            elif version == 'v5':
                self.convs.append(CGConv_v5(D_v, D_u))

        self.conv_to_fc = nn.Linear(D_v, hidden_fea_len)                        # Entry layer: D_v -> hidden_fea_len
        if self.L_hidden > 1:
            self.fcs = nn.ModuleList([nn.Linear(hidden_fea_len, hidden_fea_len) for _ in range(self.L_hidden - 1)])
        self.fc_out = nn.Linear(hidden_fea_len, out_dim)               # If classification=True, out_dim usually represents the number of classes (e.g., 2)
        self.softplus = nn.Softplus()
       
        
    def forward(self, v_fea, u_fea, edge_idx, B_idx):
        """Forward pass for CGConvNet (逆伝播用の中間変数を cache に保存する)"""
        # Step 1: Embed raw atom features
        v_fea_emb = self.embedding(v_fea)       #(N_total x D_v)全結合層に通して、モデル内部の隠れ層の次元数に変換するD_V
        self.cache['v_fea_raw'] = (v_fea)  # 元の原子特徴量を保存 (N_total x orig_D_v)  )
        x = self.embedding(v_fea)  # 埋め込み層の出力 (N_total x D_v)
        conv_inputs = []  # 各Conv層への入力を保存するリスト
        for conv in self.convs:
        # Conv層に入る前の原子特徴量を保存
            conv_inputs.append(x)
            x = conv(x, u_fea, edge_idx)        
        self.cache['v_fea_emb'] = v_fea_emb
        self.cache['u_fea'] = u_fea
        self.cache['edge_idx'] = edge_idx
        self.cache['B_idx'] = B_idx
        self.cache['conv_inputs'] = conv_inputs
        v_fea_conv = v_fea_emb
        self.cache['conv_inputs'] = []  # 各Conv層への入力を保存するリスト
        for conv in self.convs:         #各conv層の数の分だけ繰り返すcache['conv_inputs'][0] ➔ 1層目に入る直前の原子特徴量 (N_total, D_v) を保存
            self.cache['conv_inputs'].append(v_fea_conv)
            v_fea_conv = conv(v_fea_conv, u_fea, edge_idx)      
        self.cache['v_fea_conv_out'] = v_fea_conv   #for文後なので畳み込み後の原子特徴量 (N_total, D_v) を保存
        v_c_fea_list = [torch.mean(v_fea_conv[idx], dim=0, keepdim=True) for idx in B_idx]       #idx番目の結晶の原子特徴量を平均化して結晶特徴量に変換する
        v_c_fea = torch.cat(v_c_fea_list, dim=0)  # Shape: B x D_v
        self.cache['v_c_fea_pooled'] = v_c_fea  # Pooling直後のベクトル
        fc_in_z = self.conv_to_fc(v_c_fea)  # Softplusの直前の値(z)
        self.cache['fc_in_z'] = fc_in_z
        v_c_fea_act = self.softplus(fc_in_z)
        self.cache['v_c_fea_act'] = v_c_fea_act
        if self.L_hidden > 1:
            self.cache['fcs_inputs'] = []
            self.cache['fcs_z'] = []

            for fc in self.fcs:             #self/fcsに入っている層の数(L_hidden-1)だけ繰り返す
                self.cache['fcs_inputs'].append(v_c_fea_act)            #[2層目の入力, 3層目の入力] の2個が入る
                z = fc(v_c_fea_act)
                self.cache['fcs_z'].append(z)                           #[2層目のz, 3層目のz] の2個が入る
                v_c_fea_act = self.softplus(z)
        # 最終的な隠れ表現を保存（fc_out への入力）
        self.cache['last_hidden_act'] = v_c_fea_act
        out = self.fc_out(v_c_fea_act)
        self.cache['out'] = out

        return out
    
    def manual_backward(self, d_out):
        """出力 Loss から入力に向かって、連鎖律に従って手動で微分を遡っていく！"""
        out = self.cache['out']
        v_c_fea_act = self.cache['v_c_fea_act']
        fc_in_z = self.cache['fc_in_z']
        v_c_fea_pooled = self.cache['v_c_fea_pooled']
        u_fea = self.cache['u_fea']
        edge_idx = self.cache['edge_idx']
        conv_inputs = self.cache['conv_inputs']
        B_idx = self.cache['B_idx']

        B = out.shape[0]  # 最終出力B×out_dimの[0]番目の次元がバッチサイズBと定義

        # 重み W_fc_out の勾配  δT×X
        self.grads['fc_out.weight'] = torch.matmul(d_out.T, v_c_fea_act)    #out_dim×hidden_fea_len
        # バイアス b_fc_out の勾配
        self.grads['fc_out.bias'] = torch.sum(d_out, dim=0)                 #B×out_dim
        # 前の層（Softplusの出力）に送る勾配: dL / d(v_c_fea_act)   δ×W^T
        d_v_c_fea_act = torch.matmul(d_out, self.fc_out.weight)             #hidden_fea_len

        # 複数隠れ層がある場合 (L_hidden > 1)
        if self.L_hidden > 1:
            num_fcs = len(self.fcs)             #L_hidden-1
            # reversed()で後ろの層から手前へ向かって逆順にループ
            for i in reversed(range(num_fcs)):
                fc = self.fcs[i]
                x = self.cache['fcs_inputs'][i]  # 入力
                z = self.cache['fcs_z'][i]  # Softplus直前の値

                # Softplus の微分: dL/dz = (dL/dy) * sigmoid(z)
                d_z = d_v_c_fea_act * torch.sigmoid(z)

                # 重みとバイアスの勾配
                self.grads[f'fcs.{i}.weight'] = torch.matmul(d_z.T, x)  # hidden_fea_len × hidden_fea_len
                self.grads[f'fcs.{i}.bias'] = torch.sum(d_z, dim=0)     # hidden_fea_len    

                # 前の層へ渡す勾配: dL/dx = dL/dz * W^T
                d_v_c_fea_act = torch.matmul(d_z, fc.weight)

        d_fc_in_z = d_v_c_fea_act * torch.sigmoid(fc_in_z)
        self.grads['conv_to_fc.weight'] = torch.matmul(d_fc_in_z.T, v_c_fea_pooled)
        self.grads['conv_to_fc.bias'] = torch.sum(d_fc_in_z, dim=0)
        d_v_c_fea_pooled = torch.matmul(d_fc_in_z, self.conv_to_fc.weight)

        d_v_fea_conv_out = torch.zeros_like(self.cache['v_fea_conv_out'])

        for b, idx in enumerate(B_idx):
            N_b = len(idx)  # 結晶 b に含まれる原子数
            # 結晶 b の勾配を原子数 N_b で割り、所属する全原子へ均等に分配
            d_v_fea_conv_out[idx] = d_v_c_fea_pooled[b : b + 1] / N_b

        d_v_fea_conv = d_v_fea_conv_out  # Shape: (N_total, D_v)
        num_convs = len(self.convs)

        for i in reversed(range(num_convs)):
            conv = self.convs[i]
            v_fea_in = conv_inputs[i]  # i番目のConv層への入力

            # 各 CGConv 層内部の手動逆伝播を実行
            # 入力に対する勾配 d_v_fea_in と、Conv層内部パラメータの勾配辞書を取得
            d_v_fea_conv, conv_grads = conv.manual_backward(d_v_fea_conv, v_fea_in, u_fea, edge_idx)
            # 取得した勾配を全体辞書 self.grads に保存
            for param_name, grad_tensor in conv_grads.items():
                self.grads[f'convs.{i}.{param_name}'] = grad_tensor

        d_v_fea_emb = d_v_fea_conv  # Shape: (N_total, D_v)
        v_fea_raw = self.cache['v_fea_raw']  # Shape: (N_total, orig_D_v)

        self.grads['embedding.weight'] = torch.matmul(d_v_fea_emb.T, v_fea_raw)

        self.grads['embedding.bias'] = torch.sum(d_v_fea_emb, dim=0)

        return self.grads

    
