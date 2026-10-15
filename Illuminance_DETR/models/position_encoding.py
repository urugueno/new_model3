import math
import torch
from torch import nn

from util.misc import NestedTensor

class PositionEmbeddingSine1D(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        b, c, h, w = x.shape
        assert c == self.num_pos_feats, \
            f"The channel dimension {c} of the input tensor must match the num_pos_feats {self.num_pos_feats}"

        seq_len = h * w
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)

        dim_t = torch.arange(0, self.num_pos_feats, 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (dim_t / self.num_pos_feats)

        pe = torch.zeros(seq_len, self.num_pos_feats, device=x.device)
        pe[:, 0::2] = torch.sin(position / dim_t)
        pe[:, 1::2] = torch.cos(position / dim_t)

        pos = pe.T.view(1, c, h, w)
        pos = pos.repeat(b, 1, 1, 1)
        
        return pos

class PositionEmbeddingFromCoords(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, coords: torch.Tensor):
        coords = coords * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=coords.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = coords[..., 0].unsqueeze(-1) / dim_t
        pos_y = coords[..., 1].unsqueeze(-1) / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2)

class PositionEmbeddingLearned1D(nn.Module):
    def __init__(self, num_embeddings, num_pos_feats=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight)

    def forward(self, sensor_ids):
        return self.embedding(sensor_ids)

class PositionEmbeddingTime1D(nn.Module):
    def __init__(self, num_pos_feats=256, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        b, seq_len, c = x.shape
        assert c == self.num_pos_feats, \
            f"The channel dimension {c} of the input tensor must match num_pos_feats {self.num_pos_feats}"

        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, c, 2, dtype=torch.float32, device=x.device) * \
                             (-math.log(self.temperature) / c))
        
        pe = torch.zeros(seq_len, c, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0).repeat(b, 1, 1)

class PositionEmbeddingIndexSine1D(nn.Module):
    """PositionEmbeddingTime1D と同じ固定正弦波基底を、暗黙の 0..seq_len-1 ではなく
    明示的なインデックス値（例: top-k 選択後も保持される元のセンサスロット番号）に適用する版。
    センサ座標ではなくスロット番号を関数の入力とすることで、
    「一意な識別子さえあれば十分か（幾何は不要か）」を検証するための条件に使う。
    """
    def __init__(self, num_pos_feats=256, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, index: torch.Tensor):
        index = index.to(torch.float32)
        c = self.num_pos_feats

        div_term = torch.exp(torch.arange(0, c, 2, dtype=torch.float32, device=index.device) *
                             (-math.log(self.temperature) / c))

        args = index.unsqueeze(-1) * div_term
        pe = torch.zeros(*index.shape, c, device=index.device)
        pe[..., 0::2] = torch.sin(args)
        pe[..., 1::2] = torch.cos(args)

        return pe

class PositionEmbeddingLinearFromCoords(nn.Module):
    """センサ座標 (x, y) を単一の線形層で符号化する。SCE の正弦波基底が本当に必要かを見るための対照。"""
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.linear = nn.Linear(2, num_pos_feats)

    def forward(self, coords: torch.Tensor):
        return self.linear(coords[..., :2])

class PositionEmbeddingMLPFromCoords(nn.Module):
    """センサ座標 (x, y) を学習可能な小さな MLP で符号化する。
    「学習可能だが座標に接地している」中間的な条件として、
    学習可能な index 埋め込み（PositionEmbeddingLearned1D）と対比させる。
    """
    def __init__(self, num_pos_feats=256, mlp_hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_pos_feats),
        )

    def forward(self, coords: torch.Tensor):
        return self.mlp(coords[..., :2])