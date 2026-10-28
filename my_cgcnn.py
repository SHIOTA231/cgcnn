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
    def __init__(self, orig_D_v, D_u, D_v=64, t_conv=3, hidden_fea_len=128, L_hidden=1, out_dim=1, classification=False, version='v5'):   # out_dim: target dimension or num classes
        super(CGConvNet, self).__init__()
        self.version = version
        self.L_hidden = L_hidden
        self.classification = classification
        self.cache = {}  # 順伝播の中間変数を保存するよう
        self.grads = {}  # 手計算した勾配を保存するよう
        
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
        self.conv_to_fc = nn.Linear(D_v, hidden_fea_len)      # Entry layer: D_v -> hidden_fea_len
        # Multiple hidden layers if L_hidden > 1
        if self.L_hidden > 1:
            self.fcs = nn.ModuleList([nn.Linear(hidden_fea_len, hidden_fea_len) for _ in range(self.L_hidden - 1)])
        # Output layer: hidden_fea_len -> out_dim (B x out_dim)
        self.fc_out = nn.Linear(hidden_fea_len, out_dim)               # If classification=True, out_dim usually represents the number of classes (e.g., 2)
        
        self.softplus = nn.Softplus()
       
        
    def forward(self, v_fea, u_fea, edge_idx, B_idx):
        """Forward pass for CGConvNet (逆伝播用の中間変数を cache に保存する)"""
        # Step 1: Embed raw atom features
        v_fea_emb = self.embedding(v_fea)
        self.cache['v_fea_raw'] = (
            v_fea  # 埋め込み層(Linear)の逆伝播で使うため保存
        )
        self.cache['v_fea_emb'] = v_fea_emb

        # Step 2: Pass through CGConv layers
        v_fea_conv = v_fea_emb
        self.cache['conv_inputs'] = []  # 各Conv層への入力を保存するリスト

        for conv in self.convs:
            self.cache['conv_inputs'].append(v_fea_conv)
            v_fea_conv = conv(v_fea_conv, u_fea, edge_idx)

        self.cache['v_fea_conv_out'] = v_fea_conv

        # Step 3: Pooling (Mean Pooling)
        v_c_fea_list = [
            torch.mean(v_fea_conv[idx], dim=0, keepdim=True) for idx in B_idx
        ]
        v_c_fea = torch.cat(v_c_fea_list, dim=0)  # Shape: B x D_v
        self.cache['v_c_fea_pooled'] = v_c_fea  # Pooling直後のベクトル

        # Step 4: MLP layers for prediction
        fc_in_z = self.conv_to_fc(v_c_fea)  # Softplusの直前の値(z)
        self.cache['fc_in_z'] = fc_in_z

        v_c_fea_act = self.softplus(fc_in_z)
        self.cache['v_c_fea_act'] = v_c_fea_act

        # 複数隠れ層がある場合
        if self.L_hidden > 1:
            # (省略: 1層の場合は通らない)
            pass

        out = self.fc_out(v_c_fea_act)  # Shape: B x out_dim

        # ★【重要】ここを追加！ manual_backward で使うため最後に out を保存する
        self.cache['out'] = out

        return out
    
    def manual_backward(self, target):
        """出力 Loss から入力に向かって、連鎖律に従って手動で微分を遡っていく！"""
        out = self.cache['out']
        v_c_fea_act = self.cache['v_c_fea_act']
        fc_in_z = self.cache['fc_in_z']
        v_c_fea_pooled = self.cache['v_c_fea_pooled']

        B = out.shape[0]  # バッチサイズ

        # =========================================================
        # 1. 損失関数 (MSE Loss) の微分: dL / d(out)
        # =========================================================
        # L = (1/B) * sum((out - target)^2)  ==>  dL/dout = (2/B) * (out - target)
        d_out = (2.0 / B) * (out - target)

        # =========================================================
        # 2. 出力層 (fc_out: Y = X * W + b) の逆伝播
        # =========================================================
        # self.fc_out.weight の形状: (out_dim, hidden_fea_len)
        # d_out の形状: (B, out_dim)
        # v_c_fea_act の形状: (B, hidden_fea_len)

        # 重み W_fc_out の勾配
        self.grads['fc_out.weight'] = torch.matmul(d_out.T, v_c_fea_act)
        # バイアス b_fc_out の勾配
        self.grads['fc_out.bias'] = torch.sum(d_out, dim=0)

        # 前の層（Softplusの出力）に送る勾配: dL / d(v_c_fea_act)
        d_v_c_fea_act = torch.matmul(d_out, self.fc_out.weight)

        # =========================================================
        # 3. Softplus 活性化関数の逆伝播
        # =========================================================
        # Softplus: y = ln(1 + e^z)  ==>  dy/dz = 1 / (1 + e^-z) = Sigmoid(z)
        d_fc_in_z = d_v_c_fea_act * torch.sigmoid(fc_in_z)

        # =========================================================
        # 4. 全結合層 (conv_to_fc: Z = X * W + b) の逆伝播
        # =========================================================
        # 重み W_conv_to_fc の勾配
        self.grads['conv_to_fc.weight'] = torch.matmul(
            d_fc_in_z.T, v_c_fea_pooled
        )
        # バイアス b_conv_to_fc の勾配
        self.grads['conv_to_fc.bias'] = torch.sum(d_fc_in_z, dim=0)

        # 前の層（Pooling直後のベクトル）に送る勾配: dL / d(v_c_fea_pooled)
        d_v_c_fea_pooled = torch.matmul(d_fc_in_z, self.conv_to_fc.weight)

        # =========================================================
        # 5. Pooling (Mean Pooling) の逆伝播へ続く...
        # =========================================================
        # (d_v_c_fea_pooled を使って各原子の勾配へ拡張・分配していきます)

    

"""
ダミー
"""
if __name__ == "__main__":

   # 1. ハイパーパラメータの設定
    orig_atom_fea_len = 92  # atom_init.jsonの次元数
    nbr_fea_len = 41        # ガウシアン展開された距離特徴量の次元数
    atom_fea_len = 64       # 隠れ層の原子特徴量次元
    h_fea_len = 128         # MLP隠れ層の次元
    n_conv = 3              # CGConv層の数
    out_dim = 1             # 予測対象次元

    # 2. モデルのインスタンス化
    model = CGConvNet(
        orig_D_v=orig_atom_fea_len,
        D_u=nbr_fea_len,
        D_v=atom_fea_len,
        t_conv=n_conv,
        hidden_fea_len=h_fea_len,
        out_dim=out_dim,
        version='v5'
    )

    # 3. ダミーデータの作成（エッジリスト形式）
    # 2つの結晶バッチ（合計5原子: 結晶0は原子0,1,2 / 結晶1は原子3,4）
    N_total = 5   # 全原子数
    E_total = 12  # 全エッジ数（例: 全原子から伸びている合計ボンド数）

    atom_fea = torch.randn(N_total, orig_atom_fea_len) # Shape: (5, 92)
    u_fea = torch.randn(E_total, nbr_fea_len)          # Shape: (12, 41)
    
    # edge_idx は Shape: (2, 12)
    # 1行目: 受け手原子(i), 2行目: 送り手原子(j)
    edge_idx = torch.randint(0, N_total, (2, E_total))
    
    # 結晶ごとの原子割り当てインデックス
    crystal_atom_idx = [torch.tensor([0, 1, 2]), torch.tensor([3, 4])]

    # 4. forwardの実行テスト
    try:
        out = model(atom_fea, u_fea, edge_idx, crystal_atom_idx)
        print("動作確認成功！")
        print(f"出力テンソルの Shape: {out.shape}")  # torch.Size([2, 1]) になれば正常です
    except Exception as e:
        print("エラーが発生しました:")
        print(e)
