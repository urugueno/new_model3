import math
import torch
from torch import nn

from util.misc import NestedTensor
from .position_encoding import (
    PositionEmbeddingFromCoords, PositionEmbeddingTime1D, PositionEmbeddingLearned1D,
    PositionEmbeddingIndexSine1D, PositionEmbeddingLinearFromCoords, PositionEmbeddingMLPFromCoords,
)

class ConfidenceCNN(nn.Module):
    def __init__(self, num_features, sequence_length):
        super(ConfidenceCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128 * sequence_length, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class FeatureBasedSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_sensors, return_logits=False):
        super().__init__()
        
        self.sensor_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, hidden_dim)
        )

        self.score_head = nn.Linear(hidden_dim, 1)
        self.return_logits = return_logits

    def forward(self, x):
        B, N, T = x.shape

        x_reshaped = x.view(B * N, 1, T)
        feats = self.sensor_encoder(x_reshaped)
        feats = feats.view(B, N, -1)

        scores = self.score_head(feats).squeeze(-1)
        
        if self.return_logits:
            return scores
        else:
            return torch.sigmoid(scores)
    
class IlluminanceBackbone(nn.Module):
    def __init__(self, window_size, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(window_size, hidden_dim)
        self.position_embedding = PositionEmbeddingFromCoords(hidden_dim // 2)
        self.num_channels = hidden_dim

    def forward(self, samples_dict):
        illuminance = samples_dict['tensors']
        coords = samples_dict['coords']

        features = self.projection(illuminance)
        pos = self.position_embedding(coords)

        features_reshaped = features.permute(0, 2, 1).unsqueeze(-1)
        pos_reshaped = pos.permute(0, 2, 1).unsqueeze(-1)

        mask = torch.zeros((features.shape[0], features.shape[1]), dtype=torch.bool, device=features.device)

        feature_nested_tensor = NestedTensor(features_reshaped, mask)

        return [feature_nested_tensor], [pos_reshaped]

class TimeSequenceBackbone(nn.Module):
    def __init__(self, window_size, sub_window_size, stride, num_sensors, hidden_dim, args):
        super().__init__()
        self.args = args
        self.sub_window_size = sub_window_size
        self.stride = stride
        
        input_dim = sub_window_size * num_sensors
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        self.position_embedding = PositionEmbeddingTime1D(num_pos_feats=hidden_dim)
        self.num_channels = hidden_dim

        self.seq_len = math.floor((window_size - sub_window_size) / stride) + 1

    def forward(self, samples_dict):
        illuminance = samples_dict['tensors']
        B, N, W = illuminance.shape

        sliced = illuminance.unfold(2, self.sub_window_size, self.stride)

        sliced = sliced.permute(0, 2, 1, 3).flatten(2)
        
        features = self.projection(sliced)

        pos = self.position_embedding(features)
        
        features_reshaped = features.permute(0, 2, 1).unsqueeze(-1)
        pos_reshaped = pos.permute(0, 2, 1).unsqueeze(-1)
        
        SeqLen = features.shape[1]
        mask = torch.zeros((B, SeqLen, 1), dtype=torch.bool, device=features.device)
        
        feature_nested_tensor = NestedTensor(features_reshaped, mask)
        
        return [feature_nested_tensor], [pos_reshaped]
    
class TimeSensorBackbone(nn.Module):
    # 位置符号化の設計アブレーション用のモード分類。
    # 各グループは spatial_pe_generator の呼び出し方（センサIDで引くか、座標を渡すか）でまとめており、
    # forward 側の分岐と __init__ 側の生成器選択の両方で共有する。
    INDEX_BASED_MODES = ('time_sensor_index_sine', 'time_sensor_index_learned')
    COORD_BASED_MODES = ('time_sensor', 'time_sensor_no_time_pe',
                          'time_sensor_coord_linear', 'time_sensor_coord_mlp')
    NO_TEMPORAL_PE_MODES = ('time_sensor_no_time_pe', 'time_sensor_no_pe')

    def __init__(self, sub_window_size, num_sensors, hidden_dim, args, scale='full', k=8, model_mode='time_sensor'):
        super().__init__()

        self.args = args
        self.projection = nn.Linear(sub_window_size, hidden_dim)
        self.temporal_pe_generator = PositionEmbeddingTime1D(num_pos_feats=hidden_dim)
        self.spatial_pe_generator = PositionEmbeddingFromCoords(num_pos_feats=hidden_dim // 2)
        
        self.num_channels = hidden_dim
        self.scale = scale
        self.k = k
        self.num_sensors = num_sensors

        if model_mode in self.INDEX_BASED_MODES:
            sensor_ids = torch.arange(num_sensors, dtype=torch.long)
            self.register_buffer('sensor_ids', sensor_ids)

            if model_mode == 'time_sensor_index_learned':
                # センサ ID ごとに学習するPE（旧 time_sensor_wifi_like）。固定レイアウト前提の代表条件。
                self.spatial_pe_generator = PositionEmbeddingLearned1D(
                    num_embeddings=num_sensors,
                    num_pos_feats=hidden_dim
                )
            else:  # time_sensor_index_sine
                # センサ ID に固定正弦波を適用する版。座標なしで一意性だけ与えた場合の条件。
                self.spatial_pe_generator = PositionEmbeddingIndexSine1D(num_pos_feats=hidden_dim)
        elif model_mode == 'time_sensor_coord_linear':
            self.spatial_pe_generator = PositionEmbeddingLinearFromCoords(num_pos_feats=hidden_dim)
        elif model_mode == 'time_sensor_coord_mlp':
            self.spatial_pe_generator = PositionEmbeddingMLPFromCoords(num_pos_feats=hidden_dim)
        else:
            # time_sensor / time_sensor_no_time_pe / time_sensor_no_sensor_pe / time_sensor_no_pe
            self.spatial_pe_generator = PositionEmbeddingFromCoords(num_pos_feats=hidden_dim // 2)

        self.register_buffer('cached_attn_mask', None, persistent=False)

        self.model_mode = model_mode

        self.env_projection = None
        self.env_norm = None
        if self.args.environment == 'EE':
            self.env_norm = nn.LayerNorm(num_sensors)
            self.env_projection = nn.Linear(num_sensors, hidden_dim)
        
        self.confidence_cnn = None
        self.feature_selector = None
        if self.scale == 'reductionCNNconfidence':
            self.confidence_cnn = ConfidenceCNN(
                num_features=self.num_sensors,
                sequence_length=sub_window_size
            )
        elif self.scale == 'reductionAttention':
            self.feature_selector = FeatureBasedSelector(
                input_dim=sub_window_size,
                hidden_dim=32,
                num_sensors=num_sensors,
                return_logits=True
            )
        elif self.scale == 'reductionVariance':
            pass

    def forward(self, samples_dict):
        illuminance = samples_dict['tensors']
        coords = samples_dict['coords']
        B, N, W = illuminance.shape
        C = self.num_channels

        env_pe = None
        if self.args.environment == 'EE':
            env_vector = samples_dict['env_vector']
            env_pe = self.env_projection(self.env_norm(env_vector))

        selected_sensor_indices = None

        original_sensor_padding_mask = samples_dict.get('sensor_padding_mask', None)
        sensor_padding_mask = original_sensor_padding_mask
        if self.scale == 'reductionCNNconfidence':
            cnn_input_window = illuminance[:, :, :self.projection.in_features]
            cnn_input = cnn_input_window.permute(0, 2, 1)
            
            confidence_scores = self.confidence_cnn(cnn_input)

            _, top_k_indices = torch.topk(confidence_scores, self.k, dim=1)
            selected_sensor_indices = top_k_indices
            
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            illuminance = illuminance[batch_indices, top_k_indices, :]
            coords = coords[batch_indices, top_k_indices, :]

            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]

            N = self.k

        elif self.scale == 'reductionAttention':
            cnn_input_window = illuminance[:, :, :self.projection.in_features]
            
            scores = self.feature_selector(cnn_input_window)

            if original_sensor_padding_mask is not None:
                scores = scores.masked_fill(original_sensor_padding_mask, 0.0)

            top_k_scores, top_k_indices = torch.topk(scores, self.k, dim=1)
            selected_sensor_indices = top_k_indices
            
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            
            illuminance_selected = illuminance[batch_indices, top_k_indices, :]
            
            illuminance = illuminance_selected * top_k_scores.unsqueeze(-1)
            
            coords = coords[batch_indices, top_k_indices, :]

            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]

            N = self.k
            
        elif self.scale == 'reductionVariance':
            input_window = illuminance[:, :, :self.projection.in_features]
            
            variance_scores = torch.var(input_window, dim=2)
            
            _, top_k_indices = torch.topk(variance_scores, self.k, dim=1)
            selected_sensor_indices = top_k_indices
            
            batch_indices = torch.arange(B, device=illuminance.device).unsqueeze(1).expand(-1, self.k)
            illuminance = illuminance[batch_indices, top_k_indices, :]
            coords = coords[batch_indices, top_k_indices, :]

            if original_sensor_padding_mask is not None:
                sensor_padding_mask = original_sensor_padding_mask[batch_indices, top_k_indices]

            N = self.k

        sub_windows = illuminance.unfold(2, self.projection.in_features, 1)
        SeqLen = sub_windows.shape[2]

        features = self.projection(sub_windows.permute(0, 2, 1, 3))
        
        total_pe = torch.zeros_like(features)

        if self.model_mode in self.INDEX_BASED_MODES:
            if selected_sensor_indices is not None:
                current_sensor_ids = self.sensor_ids[selected_sensor_indices]
            else:
                current_sensor_ids = self.sensor_ids.unsqueeze(0).expand(B, -1)
            spatial_pe = self.spatial_pe_generator(current_sensor_ids)
            total_pe = total_pe + spatial_pe.unsqueeze(1)

        if self.model_mode in self.COORD_BASED_MODES:
            spatial_pe = self.spatial_pe_generator(coords)
            total_pe = total_pe + spatial_pe.unsqueeze(1)

        # 空間符号の形だけを変えるアブレーションなので、時間PEは
        # time_sensor_no_time_pe / time_sensor_no_pe（時間PEを意図的に消す条件）以外は一律で乗せる。
        if self.model_mode not in self.NO_TEMPORAL_PE_MODES:
            dummy_tensor_for_time = torch.zeros(B, SeqLen, C, device=illuminance.device)
            temporal_pe = self.temporal_pe_generator(dummy_tensor_for_time)
            total_pe = total_pe + temporal_pe.unsqueeze(2)

        if self.args.environment == 'EE' and env_pe is not None:
            total_pe = total_pe + env_pe.unsqueeze(1).unsqueeze(2)

        features_reshaped = features.permute(0, 3, 1, 2)
        pos_reshaped = total_pe.permute(0, 3, 1, 2)
        
        mask = torch.zeros((B, SeqLen, N), dtype=torch.bool, device=features.device)
        
        if sensor_padding_mask is not None:
            sensor_mask_expanded = sensor_padding_mask.unsqueeze(1)
            
            mask = mask | sensor_mask_expanded

        feature_nested_tensor = NestedTensor(features_reshaped, mask)
        attn_mask = None
        
        return [feature_nested_tensor], [pos_reshaped], attn_mask, selected_sensor_indices

def build_backbone(args):
    if args.model_mode == 'time':
        model = TimeSequenceBackbone(
            window_size=args.window_size,
            sub_window_size=args.sub_window_size,
            stride=args.stride,
            num_sensors=args.actual_num_sensors,
            hidden_dim=args.hidden_dim,
            args=args
        )
    elif args.model_mode in ['time_sensor', 'time_sensor_no_time_pe', 'time_sensor_no_sensor_pe', 'time_sensor_no_pe',
                              'time_sensor_index_sine', 'time_sensor_index_learned',
                              'time_sensor_coord_linear', 'time_sensor_coord_mlp']:
        model = TimeSensorBackbone(
            sub_window_size=args.sub_window_size,
            num_sensors=args.actual_num_sensors,
            hidden_dim=args.hidden_dim,
            args=args,
            scale=args.scale,
            k=args.k_neighbors,
            model_mode=args.model_mode,
        )
    
    else:
        raise ValueError(f"Unknown model_mode: {args.model_mode}")
    print(f"task : {args.task}\nmode : {args.model_mode}\nscale: {args.scale}\nenv  : {args.environment}")

    return model