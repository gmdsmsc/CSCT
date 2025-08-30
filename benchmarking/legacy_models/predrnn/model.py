import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalLSTMCell(nn.Module):
    """
    PredRNN's ST-LSTM cell with spatiotemporal memory.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        # Gates for h and c
        self.conv_xh = nn.Conv2d(in_channels + hidden_channels, hidden_channels * 7,
                                 kernel_size=kernel_size, padding=padding)

        # Memory state transition
        self.conv_m = nn.Conv2d(hidden_channels, hidden_channels * 4,
                                kernel_size=kernel_size, padding=padding)

        # Final output projection
        self.conv_last = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)

    def forward(self, x, h, c, m):
        # x: input features (B, C, H, W)
        # h, c: hidden and cell states
        # m: spatiotemporal memory

        combined = torch.cat([x, h], dim=1)
        gates = self.conv_xh(combined)

        i_x, f_x, g_x, i_m, f_m, g_m, o_x = torch.split(gates, self.hidden_channels, dim=1)

        i_x = torch.sigmoid(i_x)
        f_x = torch.sigmoid(f_x)
        g_x = torch.tanh(g_x)

        c_next = f_x * c + i_x * g_x

        m_gates = self.conv_m(m)
        i_m, f_m, g_m, o_m = torch.split(m_gates, self.hidden_channels, dim=1)

        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        g_m = torch.tanh(g_m)

        m_next = f_m * m + i_m * g_m

        o = torch.sigmoid(o_x + o_m)
        h_next = o * torch.tanh(self.conv_last(torch.cat([c_next, m_next], dim=1)))

        return h_next, c_next, m_next


class PredRNNPredictor(nn.Module):
    """
    PredRNN-based next frame predictor, drop-in replacement for ConvLSTMPredictor.
    """
    def __init__(self, hidden_dim=64, num_layers=2):
        super().__init__()

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3),  # 84->42
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, hidden_dim, 5, 2, 2),  # 42->21
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Stack of ST-LSTM cells
        self.stlstm_cells = nn.ModuleList([
            SpatioTemporalLSTMCell(hidden_dim, hidden_dim, 3)
            for _ in range(num_layers)
        ])

        # Global context LSTM (same as before)
        self.global_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=128, batch_first=True)

        self.label_embedding = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        self.y_label_embedding = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        self.global_context_projection = nn.Linear(128, hidden_dim * 21 * 21)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 32, 4, 2, 1),  # 21->42
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 42->84
            nn.LeakyReLU(),
            nn.Conv2d(16, 10, 3, 1, 1),
        )

    def forward(self, x, label_x, label_y):
        batch_size, seq_len, _, h, w = x.shape

        # Embed labels
        label_embedding = self.label_embedding(label_x.float()).view(batch_size, self.hidden_dim, 1, 1)
        y_label_embedding = self.y_label_embedding(label_y.float()).view(batch_size, self.hidden_dim, 1, 1)

        # Init hidden, cell, and memory states
        h_t = [torch.zeros(batch_size, self.hidden_dim, h // 4, w // 4, device=x.device)
               for _ in range(self.num_layers)]
        c_t = [torch.zeros_like(h_t[0]) for _ in range(self.num_layers)]
        m_t = torch.zeros_like(h_t[0])  # shared spatiotemporal memory

        pooled_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            encoded = self.spatial_encoder(frame)  # (B, hidden_dim, H/4, W/4)

            encoded = encoded + label_embedding + y_label_embedding

            h_in, c_in, m_in = encoded, None, m_t
            for i, cell in enumerate(self.stlstm_cells):
                h_t[i], c_t[i], m_in = cell(h_in, h_t[i], c_t[i], m_in)
                h_in = h_t[i]
            m_t = m_in

            pooled = torch.mean(h_t[-1], dim=[2, 3])
            pooled_features.append(pooled)

        global_features_sequence = torch.stack(pooled_features, dim=1)
        global_lstm_output, _ = self.global_lstm(global_features_sequence)
        last_step = global_lstm_output[:, -1, :]

        proj = self.global_context_projection(last_step)
        proj = proj.view(batch_size, self.hidden_dim, 21, 21)

        decoded = self.decoder(proj)  # (B, 10, 84, 84)
        output = decoded.unsqueeze(2)  # (B, 1, 10, 84, 84)

        return output
