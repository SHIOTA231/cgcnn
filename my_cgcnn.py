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

        self.cache = {}

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
        #  線形変換および活性化関数
        f_z = self.fc_f(z)  # Gate の全結合出力 (Sigmoid直前)
        s_z = self.fc_s(z)  # Core の全結合出力 (Softplus直前)

        # --- 手動逆伝播のために中間変数をキャッシュ ---
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

        # forward 時のキャッシュ取得
        z = self.cache['z']
        f_z = self.cache['f_z']
        s_z = self.cache['s_z']
        gate_out = self.cache['gate_out']
        core_out = self.cache['core_out']

        # ---------------------------------------------------------
        # Step 6 & 5: 残差接続と集約 (index_add_) の逆伝播
        # ---------------------------------------------------------
        # i_idx に集約されていた勾配を各エッジメッセージへスライス抽出mij
        d_message = d_atom_out_fea[i_idx]  # Shape: E_total x D_v

        # ---------------------------------------------------------
        # Step 4: 要素積 (\odot) の逆伝播 (相方を掛けて分岐)
        # ---------------------------------------------------------
        d_gate_out = d_message * core_out  # Shape: E_total x D_v δ_f
        d_core_out = d_message * gate_out  # Shape: E_total x D_v δ_c

        # ---------------------------------------------------------
        # Step 3: 活性化関数と全結合層 (fc_s, fc_f) の逆伝播
        # ---------------------------------------------------------
        # Core 側 (Softplus の微分: dy/dx = Sigmoid(s_z))
        d_s_z = d_core_out * torch.sigmoid(s_z)  # Shape: E_total x D_v

        # Gate 側 (Sigmoid の微分: dy/dx = gate * (1 - gate))
        d_f_z = d_gate_out * gate_out * (1.0 - gate_out)  # Shape: E_total x D_v

        # 重み・バイアスの勾配計算 (dL/dW = \delta^T * Z, dL/db = \sum \delta)
        grad_w_s = torch.matmul(d_s_z.T, z)  # Shape: D_v x (2D_v + D_u)
        grad_b_s = torch.sum(d_s_z, dim=0)  # Shape: D_v

        grad_w_f = torch.matmul(d_f_z.T, z)  # Shape: D_v x (2D_v + D_u)
        grad_b_f = torch.sum(d_f_z, dim=0)  # Shape: D_v

        # 入力 z に対する勾配 (dL/dz = \delta * W)
        d_z_core = torch.matmul(d_s_z, self.fc_s.weight)
        d_z_gate = torch.matmul(d_f_z, self.fc_f.weight)
        d_z = d_z_core + d_z_gate  # Shape: E_total x (2D_v + D_u)

        # ---------------------------------------------------------
        # Step 2: Concat の解体 (スライス)
        # ---------------------------------------------------------
        d_v_i_fea = d_z[:, : self.D_v]  # Shape: E_total x D_v
        d_v_j_fea = d_z[:, self.D_v : 2 * self.D_v]  # Shape: E_total x D_v

        # ---------------------------------------------------------
        # Step 1: 入力原子特徴量 v_in_fea への勾配合算
        # ---------------------------------------------------------
        # 1) 残差接続からの直接の勾配 (v_in_fea の微分は 1)
        d_v_in_fea = d_atom_out_fea.clone()

        # 2) v_i (i_idx) と v_j (j_idx) からの勾配を index_add_ で元の原子位置へ加算
        d_v_in_fea.index_add_(0, i_idx, d_v_i_fea)
        d_v_in_fea.index_add_(0, j_idx, d_v_j_fea)

        # 重み・バイアスの勾配を辞書にまとめる
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
        for _ in range(t_conv):     #self.convsというリストにt_convだけオブジェクトが追加。
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
        v_fea_emb = self.embedding(v_fea)       #(N_total x D_v)全結合層に通して、モデル内部の隠れ層の次元数に変換するD_V
        self.cache['v_fea_raw'] = (
            v_fea  # 埋め込み層(Linear)の逆伝播で使うため保存
        )
        self.cache['v_fea_emb'] = v_fea_emb

        # Step 2: Pass through CGConv layers
        v_fea_conv = v_fea_emb
        self.cache['conv_inputs'] = []  # 各Conv層への入力を保存するリスト

        for conv in self.convs:         #各conv層の数の分だけ繰り返すcache['conv_inputs'][0] ➔ 1層目に入る直前の原子特徴量 (N_total, D_v) を保存
            self.cache['conv_inputs'].append(v_fea_conv)
            v_fea_conv = conv(v_fea_conv, u_fea, edge_idx)      

        self.cache['v_fea_conv_out'] = v_fea_conv   #for文後なので畳み込み後の原子特徴量 (N_total, D_v) を保存
        

        # Step 3: Pooling (Mean Pooling)
        v_c_fea_list = [
            torch.mean(v_fea_conv[idx], dim=0, keepdim=True) for idx in B_idx       #idx番目の結晶の原子特徴量を平均化して結晶特徴量に変換する
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
            self.cache['fcs_inputs'] = []
            self.cache['fcs_z'] = []

            for fc in self.fcs:             #self/fcsに入っている層の数(L_hidden-1)だけ繰り返す

                # 該当層への入力を保存
                self.cache['fcs_inputs'].append(v_c_fea_act)            #[2層目の入力, 3層目の入力] の2個が入る

                z = fc(v_c_fea_act)
                self.cache['fcs_z'].append(z)                           #[2層目のz, 3層目のz] の2個が入る

                v_c_fea_act = self.softplus(z)

        # 最終的な隠れ表現を保存（fc_out への入力）
        self.cache['last_hidden_act'] = v_c_fea_act

        out = self.fc_out(v_c_fea_act)
        self.cache['out'] = out

        return out
    
    def manual_backward(self, target):
        """出力 Loss から入力に向かって、連鎖律に従って手動で微分を遡っていく！"""
        out = self.cache['out']
        v_c_fea_act = self.cache['v_c_fea_act']
        fc_in_z = self.cache['fc_in_z']
        v_c_fea_pooled = self.cache['v_c_fea_pooled']

        B = out.shape[0]  # 最終出力B×out_dimの[0]番目の次元がバッチサイズBと定義

        # =========================================================
        # 1. 損失関数 (MSE Loss) の微分: dL / d(out)
        # =========================================================
        # L = (1/B) * sum((out - target)^2)  ==>  dL/dout = (2/B) * (out - target)
        d_out = (2.0 / B) * (out - target)

        # =========================================================
        # 2. 出力層 (fc_out: Y = X * W + b) の逆伝播
        # =========================================================
        # self.fc_out.weight の形状: (out_dim, hidden_fea_len)→W
        # d_out の形状: (B, out_dim)→δ  
        # v_c_fea_act の形状: (B, hidden_fea_len)→X

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


        # =========================================================
        # 3. Softplus 活性化関数の逆伝播（プーリング直後のfc）
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
        # Step 5: Mean Pooling の逆伝播     dL/dvi=dL/dvc,b*dvc,b/dvi
        # =========================================================
        # d_v_c_fea_pooled: (B, D_v) ➔ 結晶ごとの勾配
        # d_v_fea_conv_out: (N_total, D_v) ➔ 全原子の勾配テンソルを初期化
        d_v_fea_conv_out = torch.zeros_like(self.cache['v_fea_conv_out'])

        for b, idx in enumerate(B_idx):
            N_b = len(idx)  # 結晶 b に含まれる原子数
            # 結晶 b の勾配を原子数 N_b で割り、所属する全原子へ均等に分配
            d_v_fea_conv_out[idx] = d_v_c_fea_pooled[b : b + 1] / N_b

        # =========================================================
        # Step 6: CGConv（グラフ畳み込み層群）の逆伝播
        # =========================================================
        # 最終出力の勾配からスタートして、後ろのConv層から手前へ遡る
        d_v_fea_conv = d_v_fea_conv_out  # Shape: (N_total, D_v)
        num_convs = len(self.convs)

        for i in reversed(range(num_convs)):
            conv = self.convs[i]
            v_fea_in = self.cache['conv_inputs'][i]  # i番目のConv層への入力

            # 各 CGConv 層内部の手動逆伝播を実行
            # 入力に対する勾配 d_v_fea_in と、Conv層内部パラメータの勾配辞書を取得
            d_v_fea_conv, conv_grads = conv.manual_backward(
                d_v_fea_conv, v_fea_in, u_fea, edge_idx
            )

            # 取得した勾配を全体辞書 self.grads に保存
            for param_name, grad_tensor in conv_grads.items():
                self.grads[f'convs.{i}.{param_name}'] = grad_tensor

        # =========================================================
        # Step 7: Embedding（線形埋め込み層）の逆伝播
        # =========================================================
        # 1層目のConv層から遡ってきた勾配 d_v_fea_conv が、v_fea_emb に対する勾配 d_v_fea_emb に対応
        d_v_fea_emb = d_v_fea_conv  # Shape: (N_total, D_v)
        v_fea_raw = self.cache['v_fea_raw']  # Shape: (N_total, orig_D_v)

        # 全結合層 Y = X * W^T + b の微分公式:
        # dL/dW = δ^T * X  --> Shape: (D_v, orig_D_v)
        self.grads['embedding.weight'] = torch.matmul(d_v_fea_emb.T, v_fea_raw)

        # dL/db = sum(δ, dim=0)  --> Shape: (D_v,)
        self.grads['embedding.bias'] = torch.sum(d_v_fea_emb, dim=0)

        # 逆伝播完了: 各パラメータの勾配が入った辞書を返す
        return self.grads

    

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
