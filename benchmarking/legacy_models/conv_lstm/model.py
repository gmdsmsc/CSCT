import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                             out_channels=4 * hidden_dim,
                             kernel_size=kernel_size,
                             padding=padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTMPredictor(nn.Module):
    """
    ConvLSTM-based next frame predictor with global context and self-attention for long-term memory.
    """
    def __init__(self):
        super().__init__()
        
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3),  # 84->42
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 5, 2, 2),  # 42->21
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.conv_lstm = ConvLSTMCell(64, 64, 3)
        
        # Global context LSTM
        self.global_lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        
        self.label_embedding = nn.Sequential(
            nn.Linear(10, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )

        self.y_label_embedding = nn.Sequential(
            nn.Linear(10, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )


        self.fusion_gate_conv = nn.Conv2d(64 * 2, 64, 1)

        self.global_context_projection = nn.Linear(128, 64 * 21 * 21)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 21->42
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 42->84
            nn.LeakyReLU(),
            nn.Conv2d(16, 10, 3, 1, 1),  # <--- change from 1 → 10
            # nn.Sigmoid()  # probably remove this if you want logits for classification
        )

    def forward(self, x, label_x, label_y):
        batch_size, seq_len, _, h, w = x.shape

        # Embed labels
        label_embedding = self.label_embedding(label_x.float()).view(batch_size, 64, 1, 1)
        y_label_embedding = self.y_label_embedding(label_y.float()).view(batch_size, 64, 1, 1)

        # Initialize ConvLSTM states
        h_t = torch.zeros(batch_size, 64, h // 4, w // 4, device=x.device)
        c_t = torch.zeros(batch_size, 64, h // 4, w // 4, device=x.device)

        # Process each timestep
        pooled_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (B, 1, H, W)
            encoded = self.spatial_encoder(frame)  # (B, 64, H/4, W/4)

            encoded = encoded + label_embedding + y_label_embedding

            h_t, c_t = self.conv_lstm(encoded, (h_t, c_t))

            pooled = torch.mean(h_t, dim=[2, 3])  # (B, 64)
            pooled_features.append(pooled)

        # Stack pooled features and pass through global LSTM
        global_features_sequence = torch.stack(pooled_features, dim=1)  # (B, T, 64)
        global_lstm_output, _ = self.global_lstm(global_features_sequence)

        last_step = global_lstm_output[:, -1, :]  # (B, 128)

        # Project back to spatial map
        proj = self.global_context_projection(last_step)  # (B, 64*21*21)
        proj = proj.view(batch_size, 64, 21, 21)          # (B, 64, 21, 21)

        # Decode into predicted frame
        decoded = self.decoder(proj)  # (B, 10, 84, 84)
        output = decoded.unsqueeze(2)  # (B, 1, 10, 84, 84)

        return output
